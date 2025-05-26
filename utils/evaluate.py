import torch


def acc_count(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            target = target.flatten()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct, total


def acc_evaluate(model, test_loader, device):
    correct, total = acc_count(model, test_loader, device)

    return 100.0 * correct / total
