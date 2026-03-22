import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train(model, loader, optimizer, criterion, device):

    print("Training Loop")

    model.train()

    total_loss = 0

    for images, labels in tqdm(loader):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device, criterion):

    print("Validation Loop")

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    total_loss=0.0

    with torch.no_grad():
        for images, labels in tqdm(loader):

            outputs = model(images.to(device))

            labels = labels.to(device)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels.to(device)).sum().item()

            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total_loss /= len(loader)
    accuracy = correct / total

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss":total_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }