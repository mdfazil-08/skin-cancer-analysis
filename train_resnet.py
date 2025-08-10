import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Config
DATA_DIR = '/home/fazil/Projects/cv-projects/dataset'
CHECKPOINT_DIR = '/home/fazil/Projects/cv-projects/checkpoints'
NUM_CLASSES = 7
BATCH_SIZE = 4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(DEVICE)}")
else:
    print("Running on CPU")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Transforms with augmentation
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Datasets
datasets_ = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), transform=transform[x])
    for x in ['train', 'val', 'test']
}
dataloaders = {
    x: DataLoader(datasets_[x], batch_size=BATCH_SIZE, shuffle=True)
    for x in ['train', 'val', 'test']
}
class_names = datasets_['train'].classes

# Load or define model
def load_model_from_checkpoint():
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
    if files:
        latest = files[-1]
        print(f"Loading model from checkpoint: {latest}")
        
        # Load checkpoint directly onto the correct device
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)

        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)  # Ensure model is on correct device after loading weights

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state tensors to the right device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)

        epoch = checkpoint['epoch']
        return model, optimizer, epoch, checkpoint['loss']
    
    else:
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model.to(DEVICE), optim.Adam(model.parameters(), lr=0.0001), 0, float('inf')


# Training function
def train():
    print("Training started...")
    model, optimizer, start_epoch, best_loss = load_model_from_checkpoint()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0

        for inputs, labels in dataloaders['train']:
            torch.cuda.empty_cache()
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloaders['train'])

        # Print loss every epoch
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    return model


# Validation function
def validate(model):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f'Validation Accuracy: {acc:.2f}%')

# Inference
def inference(model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    print("\nMetrics (per class):")
    for i, name in enumerate(class_names):
        print(f"Class: {name}")
        print(f"  TP: {TP[i]}, FP: {FP[i]}, TN: {TN[i]}, FN: {FN[i]}")
        precision = TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] > 0 else 0
        recall    = TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] > 0 else 0
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}")

if __name__ == '__main__':
    model = train()
    inference(model)




