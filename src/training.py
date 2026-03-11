import torch

def train(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            outputs = model(images.to(device))

            preds = outputs.argmax(dim=1)

            correct += (preds == labels.to(device)).sum().item()

            total += labels.size(0)

    return correct / total