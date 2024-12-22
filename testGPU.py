import os
import time
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
import timm

# Set environment variable to manage CUDA memory (optional on some platforms)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
    
    table = PrettyTable(["Dataset", "Total Images", "Classes"])
    table.add_row(["Train", len(train_dataset), len(train_dataset.classes)])
    table.add_row(["Test", len(test_dataset), len(test_dataset.classes)])
    print(table)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader

def select_model(model_name, num_classes):
    print(f"Selecting model: {model_name}")
    
    if model_name == "InceptionV3":
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    return model

def train_and_evaluate(data_dir, model_name, num_classes=4, epochs=10, batch_size=24, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for {model_name}")

    img_size = (299, 299) if model_name == "InceptionV3" else (224, 224)
    train_loader, test_loader = load_data(data_dir, img_size=img_size, batch_size=batch_size)
    
    model = select_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - {model_name}", leave=False)
        start_time = time.time()

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = criterion(logits, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), accuracy=correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {elapsed_time:.2f}s")

    torch.save(model.state_dict(), f"model/{model_name}_best_model.pth")
    print(f"Model saved as {model_name}_best_model.pth")

    # Evaluation
    model.eval()
    all_preds, all_labels, total_loss, correct, total = [], [], 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{model_name} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(all_labels, all_preds))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["CNV", "DME", "DRUSEN", "NORMAL"], yticklabels=["CNV", "DME", "DRUSEN", "NORMAL"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    return accuracy, elapsed_time

# Running different models and comparing results
if __name__ == "__main__":
    data_dir = r"C:\Users\RGON\Documents\models_train\data"
    results = {}
    best_model, best_accuracy, best_time = None, 0, float('inf')
    
    for model_name in ["InceptionV3"]:
        print(f"\nTraining and Evaluating {model_name}...\n")
        accuracy, training_time = train_and_evaluate(data_dir, model_name)
        results[model_name] = (accuracy, training_time)

        if accuracy > best_accuracy or (accuracy == best_accuracy and training_time < best_time):
            best_model, best_accuracy, best_time = model_name, accuracy, training_time

    # Display summary of results
    print("\nModel Comparison Results:")
    result_table = PrettyTable(["Model", "Accuracy", "Training Time (s)"])
    for model, (acc, time) in results.items():
        result_table.add_row([model, f"{acc:.4f}", f"{time:.2f}"])
    print(result_table)

    print(f"\nBest Model: {best_model} with Accuracy: {best_accuracy:.4f} and Training Time: {best_time:.2f} seconds")
