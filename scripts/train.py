
# scripts/train.py

import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

############################################################
#                 MODEL DEFINITIONS
############################################################

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class DeepfakeDetector(nn.Module):
    def __init__(self, lstm_hidden=256, lstm_layers=1):
        super().__init__()
        self.cnn = CNNFeatureExtractor()

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        features = []

        for t in range(T):
            frame = x[:, t, :, :, :]
            feat = self.cnn(frame)
            features.append(feat)

        features = torch.stack(features, dim=1)
        _, (h, _) = self.lstm(features)
        out = self.classifier(h[-1])
        return out


############################################################
#                     DATASET
############################################################

class DeepfakeSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=30, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        self.samples = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        if os.path.isdir(real_dir):
            for f in sorted(os.listdir(real_dir)):
                self.samples.append((os.path.join(real_dir, f), 0))

        if os.path.isdir(fake_dir):
            for f in sorted(os.listdir(fake_dir)):
                self.samples.append((os.path.join(fake_dir, f), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_path, label = self.samples[idx]

        frame_files = sorted(os.listdir(folder_path))

        # ✅ SAFETY CHECK: Handle empty folders
        if len(frame_files) == 0:
            # Recursively fetch next valid sample
            return self.__getitem__((idx + 1) % len(self.samples))

        N = len(frame_files)

        if N >= self.sequence_length:
            indices = np.linspace(0, N - 1, self.sequence_length, dtype=int)
        else:
            indices = list(range(N)) + [N - 1] * (self.sequence_length - N)

        images = []
        for i in indices:
            img_path = os.path.join(folder_path, frame_files[i])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images)
        return images, torch.tensor(label).long()


############################################################
#                 METRIC CALCULATOR
############################################################

def calc_metrics(y_true, y_prob):
    y_pred = (np.array(y_prob)[:,1] >= 0.5).astype(int)
    y_true = np.array(y_true)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, np.array(y_prob)[:,1])
    except:
        auc = 0.0

    return acc, prec, rec, f1, auc


############################################################
#                    TRAIN LOOP
############################################################

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    y_true, y_prob = [], []

    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device).float(), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()

        for i in range(len(y)):
            y_true.append(y.cpu().numpy()[i])
            y_prob.append(probs[i])

    return np.mean(losses), calc_metrics(y_true, y_prob)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true, y_prob = [], []

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Val", leave=False):
            X, y = X.to(device).float(), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            losses.append(loss.item())
            probs = torch.softmax(logits.cpu(), dim=1).numpy()

            for i in range(len(y)):
                y_true.append(y.cpu().numpy()[i])
                y_prob.append(probs[i])

    return np.mean(losses), calc_metrics(y_true, y_prob)


############################################################
#                     MAIN TRAINING
############################################################

def main():
    frames_dir = "/content/drive/MyDrive/deepfake-detection/data/frames"
    save_dir = "/content/drive/MyDrive/deepfake-detection/models"

    epochs = 15
    batch_size = 4
    seq_len = 30
    lr = 1e-4
    num_workers = 2
    force_cpu = False

    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    print("Using device:", device)

    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    dataset = DeepfakeSequenceDataset(frames_dir, sequence_length=seq_len, transform=transform)

    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    s = int(0.8 * n)

    from torch.utils.data import Subset
    train_set = Subset(dataset, indices[:s])
    val_set   = Subset(dataset, indices[s:])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = DeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        _, _, _, train_f1, _ = train_metrics
        _, _, _, val_f1, _ = val_metrics

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print("Saved Best Model →", save_path)

    print("\nTraining Finished.")
    print("Best F1 Score:", best_val_f1)


if __name__ == "__main__":
    main()
