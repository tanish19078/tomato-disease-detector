"""
Tomato Leaf Disease Detection — EfficientNet-B0 Training Script (FIXED)
========================================================================
Fix: Uses a TransformSubset wrapper so train/val/test each get their own
     transforms, instead of mutating the shared dataset object.

Run on Google Colab with T4 GPU:
  1. Upload the 5 zip files to /content/
  2. Run this script
  3. Download the .onnx, .onnx.data, and class_mapping.json files
"""

import os
import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
from torchvision import datasets, transforms, models

print("PyTorch Version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# ─── 1. Dataset Setup ────────────────────────────────────────────────────

# === FIX: Wrapper class that applies its OWN transform to a Subset ===
class TransformSubset(Dataset):
    """
    Wraps a torch.utils.data.Subset with a custom transform.
    This avoids the bug where setting .dataset.transform mutates
    the shared parent dataset for ALL subsets.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        # subset returns already-transformed data if parent has a transform,
        # so we need raw images. We access the underlying dataset directly.
        raw_img, label = self.subset.dataset.samples[self.subset.indices[index]]
        from PIL import Image
        img = Image.open(raw_img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# Unzip data if needed
data_dir = "/content/dataset"
if not os.path.exists(data_dir):
    import glob, zipfile
    os.makedirs(data_dir)
    print("Unzipping dataset files...")
    for zip_file in glob.glob("/content/*.zip"):
        print(f"  Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    print("Finished unzipping!")
else:
    print("Dataset directory already exists. Skipping extraction.")


input_size = 224
batch_size = 32

train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset WITHOUT transform — raw images
full_dataset = datasets.ImageFolder(data_dir, transform=None)

# 80/10/10 split
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(42)
)

# === FIX: Wrap each subset with its own transform ===
train_data = TransformSubset(train_subset, train_transform)
val_data = TransformSubset(val_subset, val_transform)
test_data = TransformSubset(test_subset, val_transform)

dataloaders = {
    'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2),
    'test': DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
}
dataset_sizes = {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}
class_names = full_dataset.classes

print(f"Classes: {class_names}")
print(f"Splits: {dataset_sizes}")


# ─── 2. Model Definition ─────────────────────────────────────────────────

def initialize_model(num_classes, feature_extract=False):
    model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    if feature_extract:
        for param in model_ft.parameters():
            param.requires_grad = False

    # Custom classifier head with MC Dropout
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    return model_ft


model_ft = initialize_model(num_classes=len(class_names), feature_extract=False)
model_ft = model_ft.to(device)


# ─── 3. Training Configuration ───────────────────────────────────────────

optimizer_ft = optim.AdamW(model_ft.parameters(), lr=5.97e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# ─── 4. Training Loop ────────────────────────────────────────────────────

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)

            print(f'  {phase:5s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'  *** New best val acc: {best_acc:.4f} ***')
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history


NUM_EPOCHS = 15
model_ft, t_acc, v_acc, t_loss, v_loss = train_model(
    model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS
)

torch.save(model_ft.state_dict(), "best_efficientnet_tomato.pth")
print("Saved best_efficientnet_tomato.pth")


# ─── 5. Plot Training Curves ─────────────────────────────────────────────

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_acc, label="Train Acc", marker='o', markersize=3)
plt.plot(v_acc, label="Val Acc", marker='s', markersize=3)
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t_loss, label="Train Loss", marker='o', markersize=3)
plt.plot(v_loss, label="Val Loss", marker='s', markersize=3)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle("EfficientNet-B0 — Tomato Disease Detection (Fixed Augmentation)")
plt.tight_layout()
plt.savefig("efficientnet_b0_training_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved training curves plot.")


# ─── 6. Test Set Evaluation ──────────────────────────────────────────────

model_ft.eval()
test_corrects = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)
        test_total += labels.size(0)

test_acc = test_corrects.double() / test_total
print(f"\nTest Set Accuracy: {test_acc:.4f} ({test_corrects}/{test_total})")


# ─── 7. ONNX Export ──────────────────────────────────────────────────────

model_ft.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model_ft,
    dummy_input,
    "tomato_disease_efficientnet.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Save class mapping
with open("class_mapping.json", "w") as f:
    json.dump({k: v for k, v in enumerate(class_names)}, f)

print("\nExported to ONNX successfully!")
print("Files to download:")
print("  - tomato_disease_efficientnet.onnx")
print("  - tomato_disease_efficientnet.onnx.data")
print("  - class_mapping.json")
print("  - best_efficientnet_tomato.pth")
print("  - efficientnet_b0_training_curves.png")
