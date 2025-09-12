import torch 
import torch.nn as nn

def evaluate(model, testloader, device,type='ann'):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs = inputs.to(device)
            if type == 'snn':
                outputs = model(inputs).mean(0)
            else:
                outputs = model(inputs)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        final_acc = 100 * correct / total
    return final_acc

