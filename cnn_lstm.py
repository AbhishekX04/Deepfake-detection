import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMDetector(nn.Module):
    def __init__(self, pretrained=True, sequence_length=30):
        super().__init__()
        base = models.mobilenet_v2(pretrained=pretrained)
        self.cnn = base.features
        
        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True       # IMPORTANT!
        )
        
        # Because bidirectional=True → output size = hidden_size*2 = 512
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        x = x.reshape(B*T, C, H, W)
        feats = self.cnn(x)
        feats = feats.mean(dim=[2,3])          # (B*T, 1280)
        feats = feats.reshape(B, T, -1)        # (B, T, 1280)

        lstm_out, _ = self.lstm(feats)
        
        out = self.classifier(lstm_out[:, -1])  # last time step
        return out
