pip install torch torchvision

import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageFilter
from torchvision import transforms, models
from captum.attr import IntegratedGradients, LayerGradCam

# --- Define your MoCo model and load weights (as defined earlier) ---
class MoCo(nn.Module):
    def __init__(self, base_model, feature_dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        super(MoCo, self).__init__()
        self.encoder_q = base_model(num_classes=feature_dim)
        self.encoder_k = base_model(num_classes=feature_dim)
        self.temperature = temperature
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def forward(self, x_q, x_k):
        q = self.encoder_q(x_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            k = self.encoder_k(x_k)
            k = nn.functional.normalize(k, dim=1)
        return q, k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoCo(models.resnet18, feature_dim=128).to(device)
checkpoint_path = "checkpoint_epoch_48.pth"  # Update with your checkpoint path
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    st.write("✅ Loaded checkpoint")
else:
    st.write("🚩 No checkpoint found. Using randomly initialized weights.")

# --- Define your Integrated Gradients explanation function ---
def explain_moco_ig(moco_model, image, device, n_steps=50, baseline_type="black"):
    moco_model.eval()
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    # Create a baseline using a black image
    if baseline_type == "black":
        baseline_image = Image.new("RGB", image.size, (0, 0, 0))
    elif baseline_type == "white":
        baseline_image = Image.new("RGB", image.size, (255, 255, 255))
    elif baseline_type == "blur":
        baseline_image = image.filter(ImageFilter.GaussianBlur(radius=10))
    elif baseline_type == "mean":
        baseline_image = Image.new("RGB", image.size, (123, 116, 103))
    else:
        baseline_image = Image.new("RGB", image.size, (0, 0, 0))
    baseline = test_transform(baseline_image).unsqueeze(0).to(device)
    
    def forward_func(x):
        features = moco_model.encoder_q(x)
        return features.sum(dim=1, keepdim=True)
    
    ig = IntegratedGradients(forward_func)
    attr, _ = ig.attribute(input_tensor, baseline, target=0, n_steps=n_steps, return_convergence_delta=True)
    attr = attr.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    heatmap = np.mean(np.abs(attr), axis=2)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    image_np = np.array(image)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * image_np + 0.5 * heatmap_color).astype(np.uint8)
    return overlay

# --- Streamlit App Interface ---
st.title("MoCo Model XAI Demo")
st.write("Upload an image to see the Integrated Gradients explanation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
baseline_type = st.selectbox("Choose a baseline type", ["black", "white", "blur", "mean"])

if uploaded_file is not None:
    test_image = Image.open(uploaded_file).convert("RGB")
    st.image(test_image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Computing explanation..."):
        overlay = explain_moco_ig(model, test_image, device, n_steps=50, baseline_type=baseline_type)
    
    st.image(overlay, caption="XAI Explanation Overlay", use_column_width=True)

