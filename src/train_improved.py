# src/train_improved.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shutil

TRAIN_DIR = r"d:\AlfaStack Assignment\data\train"
TEST_DIR  = r"d:\AlfaStack Assignment\data\test"
MODELS_DIR = r"d:\AlfaStack Assignment\models"
PLOTS_DIR  = r"d:\AlfaStack Assignment\plots"
RESULTS_DIR = r"d:\AlfaStack Assignment\results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

test_transform = val_transform

full_train_for_split = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)  # for splitting (uses labels)
classes = full_train_for_split.classes
print("Classes:", classes)

targets = full_train_for_split.targets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(TRAIN_DIR, transform=val_transform) 

train_subset = Subset(train_dataset, train_idx)
val_subset   = Subset(val_dataset, val_idx)

train_labels = [train_dataset.targets[i] for i in train_idx]
class_sample_count = np.array([train_labels.count(i) for i in range(len(classes))])

class_sample_count = np.maximum(class_sample_count, 1)
weights_per_class = 1.0 / class_sample_count
samples_weight = np.array([weights_per_class[label] for label in train_labels])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

BATCH_SIZE = 32
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Train samples:", len(train_subset), "Val samples:", len(val_subset), "Test samples:", len(test_dataset))

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))
model = model.to(device)

counts = np.array([train_labels.count(i) for i in range(len(classes))], dtype=float)
class_weights = torch.tensor((counts.sum() / (counts + 1e-8)), dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

params_to_update = [
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
]
optimizer = optim.SGD(params_to_update, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

EPOCHS = 20
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}: Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = os.path.join(MODELS_DIR, "resnet18_best_finetuned.pth")
        torch.save(model.state_dict(), best_path)
        print("  -> Best model saved:", best_path)

    scheduler.step()

print("Training finished. Best val acc:", best_val_acc)

best_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
best_model.fc = nn.Linear(best_model.fc.in_features, len(classes))
best_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet18_best_finetuned.pth"), map_location=device))
best_model = best_model.to(device)
best_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nTest Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.title("Confusion matrix (test)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_test.png"))
plt.close()
print("Saved confusion matrix:", os.path.join(PLOTS_DIR, "confusion_matrix_test.png"))

MIS_DIR = os.path.join(RESULTS_DIR, "misclassified")
shutil.rmtree(MIS_DIR, ignore_errors=True)
os.makedirs(MIS_DIR, exist_ok=True)

test_paths = test_dataset.samples  
mistakes_saved = 0
for idx, (path, true_label) in enumerate(test_paths):
    pred = all_preds[idx]
    if pred != true_label:
        dest = os.path.join(MIS_DIR, f"true_{classes[true_label]}__pred_{classes[pred]}__{os.path.basename(path)}")
        try:
            Image.open(path).convert("RGB").save(dest)
            mistakes_saved += 1
        except:
            pass
print(f"Saved {mistakes_saved} misclassified images to {MIS_DIR}")

plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png")); plt.close()

plt.figure()
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "acc_curve.png")); plt.close()

print("Saved training curves to", PLOTS_DIR)
print("✅ All done. Check results in:", RESULTS_DIR, PLOTS_DIR, MODELS_DIR)
