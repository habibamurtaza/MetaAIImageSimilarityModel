# model/moco.py
import torch
import torch.nn as nn
from torchvision import models

class MoCo(nn.Module):
    def __init__(self, base_model=models.resnet18, feature_dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
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

class MoCoLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MoCoLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, q, k, queue):
        batch_size = q.shape[0]
        logits_pos = torch.matmul(q, k.T) / self.temperature
        logits_neg = torch.matmul(q, queue.clone().detach()) / self.temperature
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        labels = torch.arange(batch_size, device=q.device)
        return self.criterion(logits, labels)
