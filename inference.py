# model/inference.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from captum.attr import IntegratedGradients, LayerGradCam

def get_baseline_tensor(image, transform, baseline_type="black"):
    if baseline_type == "black":
        baseline_image = Image.new("RGB", image.size, (0, 0, 0))
    elif baseline_type == "white":
        baseline_image = Image.new("RGB", image.size, (255, 255, 255))
    elif baseline_type == "blur":
        baseline_image = image.filter(ImageFilter.GaussianBlur(radius=10))
    elif baseline_type == "mean":
        baseline_image = Image.new("RGB", image.size, (123, 116, 103))
    else:
        raise ValueError("Unknown baseline_type. Use 'black', 'white', 'blur', or 'mean'.")
    baseline_tensor = transform(baseline_image).unsqueeze(0)
    return baseline_tensor

def explain_moco_ig(model, image, device, n_steps=50, baseline_type="black", reference_path=None):
    """
    Generates an overlay image with an attribution heatmap using Integrated Gradients.
    """
    model.eval()
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    
    if reference_path:
        try:
            ref_image = Image.open(reference_path).convert("RGB")
            baseline = get_baseline_tensor(ref_image, test_transform, baseline_type=baseline_type).to(device)
        except Exception as e:
            print("Error loading reference image, using default baseline.", e)
            baseline = get_baseline_tensor(image, test_transform, baseline_type=baseline_type).to(device)
    else:
        baseline = get_baseline_tensor(image, test_transform, baseline_type=baseline_type).to(device)
        
    def forward_func(x):
        features = model.encoder_q(x)
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

def gradcam_explanation(model, image, device, target_layer, target=0):
    """
    Computes GradCAM attribution for a chosen target layer.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    layer_grad_cam = LayerGradCam(model.encoder_q, target_layer)
    attr = layer_grad_cam.attribute(input_tensor, target=target, relu_attributions=True)
    attr_upsampled = torch.nn.functional.interpolate(attr, size=(image.size[1], image.size[0]),
                                                       mode='bilinear', align_corners=False)
    attr_upsampled = attr_upsampled.squeeze().cpu().detach().numpy()
    heatmap = (attr_upsampled - attr_upsampled.min()) / (attr_upsampled.max() - attr_upsampled.min() + 1e-8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    image_np = np.array(image)
    overlay = (0.5 * image_np + 0.5 * heatmap_color).astype(np.uint8)
    return overlay, heatmap
