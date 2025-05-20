import os
import random
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from torchvision.models.resnet import BasicBlock, ResNet

class Cutout(object):
    def __init__(self, n_holes, length):
        """
        n_holes: Number of images for cutout
        length: Cutout length
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Apply CutOut augmentation on the image
        """
        img = img.clone()
        h, w = img.shape[1], img.shape[2]

        for _ in range(self.n_holes):
            y = random.randint(0, h - self.length)
            x = random.randint(0, w - self.length)

            img[:, y:y + self.length, x:x + self.length] = 0
        return img


if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser('A_level', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    # Dataset
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # Train
    parser.add_argument('--max_epochs', default=50, type=int)
    # Result
    parser.add_argument('--result_dir', default='./results/', type=str)
    args = parser.parse_args()

    # Load configurations
    seed = args.seed
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    result_dir = args.result_dir
    #  Set up initial environment
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    # Set transforms
    # Training set transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=8),
        transforms.Normalize(mean, std),
    ])
    # Validation and test set transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    
    # Get dataset
    train_val_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    # Split dataset
    train_indices, val_indices = random_split(range(len(train_val_set)), [49500, 500])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=None)
    train_set = Subset(train_dataset, train_indices)

    val_dataset = datasets.CIFAR100(root='./data', train=True, download=False, transform=test_transform)
    val_set = Subset(val_dataset, val_indices)


    # # Get dataset -  CiFAR10
    # train_val_set = datasets.CIFAR10(root='./try_dataset', train=True, download=True, transform=None)
    # test_set = datasets.CIFAR10(root='./try_dataset', train=False, download=True, transform=test_transform)
    # # Split dataset
    # train_indices, val_indices = random_split(range(len(train_val_set)), [49500, 500])
    # train_dataset = datasets.CIFAR10(root='./try_dataset', train=True, download=False, transform=None)
    # train_set = Subset(train_dataset, train_indices)

    # val_dataset = datasets.CIFAR10(root='./try_dataset', train=True, download=False, transform=test_transform)
    # val_set = Subset(val_dataset, val_indices)
    
    
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = test_transform
    # Define data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Set result folder
    result_folder = args.result_dir
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print("Result_folder created: ", result_folder)
    with open(result_folder + 'args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')


    # Model - baseline resnet18
    num_classes = 100 # modify it based on the import dataset of CIFAR10 or CIFAR100
    model = resnet18(weights=None)
    # # model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )


    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epochs)

    # Initialize metrics
    loss_history_train = []
    loss_history_val = []
    acc_history_val = []
    best_acc = -1
    best_metric_epoch = -1
    best_params = None
    # Train
    for epoch in range(max_epochs):
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        for idx, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"epoch {epoch + 1}/{max_epochs}",
                  f"lr: {optimizer.param_groups[0]['lr']:.4e}",
                  f"{step}/{len(train_set) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_set) // train_loader.batch_size
        scheduler.step()
        epoch_loss /= step
        loss_history_train.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Initialize validation metrics
        step = 0
        loss_val = 0
        # Validating
        with torch.no_grad():
            model.eval()
            group_pred = []
            group_label = []
            for idx, val_data in enumerate(val_loader):
                step += 1
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                output = model(val_images)
                loss = criterion(output, val_labels)
                loss_val += loss.item()

                probabilities = F.softmax(output, dim=1)
                pred_output_cpu = torch.argmax(probabilities, dim=1).cpu().numpy()
                val_labels_cpu = val_labels.cpu().numpy()

                for i in range(len(pred_output_cpu)):
                    group_pred.append(pred_output_cpu[i])
                    group_label.append(val_labels_cpu[i])

        loss_val /= step
        loss_history_val.append(loss_val)

        acc_val = accuracy_score(group_label, group_pred)
        acc_history_val.append(acc_val)

        print(
            f"Validation results:"
            f"Average Accuracy: {acc_val:.4f} "
            f"Validation Loss: {loss_val:.4f}"
        )

        if acc_val > best_acc:
            best_acc = acc_val
            best_metric_epoch = epoch + 1
            best_params = model.state_dict()
            torch.save(model.state_dict(), os.path.join(result_folder, "best_metric_model.pth"))
            print("Saved new best metric model")

    # Test
    model.load_state_dict(best_params)
    # Initialize test metrics
    loss_test = 0
    step = 0
    # Testing
    with torch.no_grad():
        model.eval()
        group_pred = []
        group_label = []
        for idx, test_data in enumerate(test_loader):
            step += 1
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
            output = model(test_images)
            loss = criterion(output, test_labels)
            loss_test += loss.item()

            probabilities = F.softmax(output, dim=1)
            pred_output_cpu = torch.argmax(probabilities, dim=1).cpu().numpy()
            test_labels_cpu = test_labels.cpu().numpy()

            for i in range(len(pred_output_cpu)):
                group_pred.append(pred_output_cpu[i])
                group_label.append(test_labels_cpu[i])

    loss_test /= step
    acc_test = accuracy_score(group_label, group_pred)

    print(
        f"Test results:"
        f"Average Accuracy: {acc_test:.4f} "
        f"Validation Loss: {loss_test:.4f}"
    )

    plt.clf()
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(max_epochs), loss_history_train, label="Train Loss")
    plt.plot(range(max_epochs), loss_history_val, label="Val Loss")
    plt.legend()
    plt.savefig(os.path.join(result_folder, "loss_history_50.png"))

    plt.clf()
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(max_epochs), acc_history_val, label="Val Accuracy")
    plt.savefig(os.path.join(result_folder, "acc_history_50.png"))
