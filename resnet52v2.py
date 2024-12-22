import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Set environment variable to help manage CUDA memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_data(data_dir, img_size=(224, 224), batch_size=16):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetClassifier, self).__init__()
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

def train_model(data_dir, num_classes=4, epochs=10, batch_size=16, device="cuda", accumulation_steps=2):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_data(data_dir, img_size=(224, 224), batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = MobileNetClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create models directory if it does not exist
    os.makedirs("models", exist_ok=True)

    # Training loop with gradient accumulation
    model.train()
    scaler = torch.cuda.amp.GradScaler()  # Corrected API usage
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()
        epoch_start_time = time.time()  # Start time for epoch

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():  # Corrected API usage
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_accuracy = correct / total
        epoch_time = time.time() - epoch_start_time  # Calculate epoch time

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {epoch_accuracy:.4f}, Time: {epoch_time:.2f}s")

    torch.save(model.state_dict(), "models/best_mobilenet_model.pth")
    print("Model saved to models/best_mobilenet_model.pth")

def eval_model(model_path, data_dir, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = load_data(data_dir, img_size=(224, 224), batch_size=16)

    model = MobileNetClassifier(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    data_dir = r"C:\Users\RGON\Documents\models_train\data"
    
    # Train the model
    train_model(data_dir)
    
    # Evaluate the model
    model_path = "models/best_mobilenet_model.pth"
    eval_model(model_path, data_dir)
