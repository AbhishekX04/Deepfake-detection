import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, class_score):
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(1, 2))

        cam = torch.zeros(self.activations.shape[1:], device=self.activations.device)

        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()

        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
