import torch

def test_model(model, data_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            inputs = batch.text
            labels = batch.label-1
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Predictions and accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # Compute average loss and accuracy
    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions * 100

    return average_loss, accuracy