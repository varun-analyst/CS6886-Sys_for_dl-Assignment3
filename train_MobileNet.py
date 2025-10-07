import torch
import torch.nn as nn
import torch.nn.functional as F
# from VGG import vgg16
from mobilenetv2 import mobilenet_v2
# from dataloader import get_cifar10

from torch.utils.tensorboard import SummaryWriter

from dataloader import get_cifar100

from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(log_dir='runs/cifar100_mobilenetv2')

if __name__ == "__main__":

    train_loader,test_loader = get_cifar100(batchsize=256)

    # model = vgg16(num_classes=10, dropout=0.5)

    model = mobilenet_v2(num_classes=100, dropout=0.3)
    model.to(device)

    epochs = 800
    warmup_epochs = 10

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05,momentum=0.9,weight_decay=5e-4,nesterov=True)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[warmup_scheduler, cosine_scheduler], 
    milestones=[warmup_epochs]
)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()
            loss.mean().backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader, device)

        writer.add_scalar('Loss/train', train_loss, epoch)

        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', test_acc, epoch)


        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")
        scheduler.step()
    
    writer.close()
    # torch.save(model.state_dict(),'./checkpoints/test2_vgg16_cifar10.pth')
    torch.save(model.state_dict(),'./checkpoints/mobilenetv2_cifar10.pth')