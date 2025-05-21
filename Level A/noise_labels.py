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
from torch.utils.data import DataLoader, random_split, Subset, Dataset

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# New import for ImageNette
from fastai.vision.all import *


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


class NoisyDataset(Dataset):
    def __init__(self, subset, noisy_labels, original_labels=None, transform=None):
        self.subset = subset
        self.noisy_labels = noisy_labels
        self.original_labels = original_labels if original_labels is not None else [subset[i][1] for i in
                                                                                    range(len(subset))]
        self.transform = transform

    def __getitem__(self, index):
        x, _ = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, self.noisy_labels[index], self.original_labels[index]

    def __len__(self):
        return len(self.noisy_labels)


def add_noise_to_labels(labels, noise_level=0.2, num_classes=10, noise_type='uniform'):
    """Add different types of label noise"""
    noisy_labels = labels.copy()
    mask = np.random.rand(len(labels)) < noise_level

    if noise_type == 'uniform':
        noisy_labels[mask] = np.random.randint(0, num_classes, size=mask.sum())
    elif noise_type == 'flip':
        # Flip to next class (more realistic noise)
        noisy_labels[mask] = (labels[mask] + 1) % num_classes
    return noisy_labels


class CoTeaching:
    """Co-teaching method for learning with noisy labels"""

    def __init__(self, model1, model2, optimizer1, optimizer2, criterion, forget_rate=0.2):
        self.model1 = model1
        self.model2 = model2
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.criterion = criterion
        self.forget_rate = forget_rate

    def update(self, inputs, labels):
        # Forward pass
        outputs1 = self.model1(inputs)
        outputs2 = self.model2(inputs)

        # Calculate per-sample loss
        loss1 = self.criterion(outputs1, labels)
        loss2 = self.criterion(outputs2, labels)

        # Select small-loss instances
        remember_rate = 1 - self.forget_rate
        num_remember = int(remember_rate * len(inputs))

        # Ensure num_remember is within valid range
        num_remember = max(1, min(num_remember, len(inputs)))

        # Now use topk safely
        _, indices1 = torch.topk(loss1, num_remember, largest=False)
        _, indices2 = torch.topk(loss2, num_remember, largest=False)

        # Update models with selected samples
        self.optimizer1.zero_grad()
        loss1_update = self.criterion(outputs1[indices2], labels[indices2]).mean()  # Added .mean()
        loss1_update.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2_update = self.criterion(outputs2[indices1], labels[indices1]).mean()  # Added .mean()
        loss2_update.backward()
        self.optimizer2.step()

        return loss1_update.item(), loss2_update.item()


def create_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser('Noisy Labels', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--noise_level', default=0.3, type=float, help='Fraction of labels to corrupt')
    parser.add_argument('--noise_type', default='uniform', choices=['uniform', 'flip'], help='Type of label noise')
    parser.add_argument('--method', default='coteaching', choices=['standard', 'coteaching'],
                        help='Method to handle noisy labels')
    parser.add_argument('--result_dir', default='./results/', type=str)
    args = parser.parse_args()

    # Load configurations
    seed = args.seed
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    noise_level = args.noise_level
    noise_type = args.noise_type
    method = args.method
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
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize(128),  # ImageNette images are larger than CIFAR
        transforms.CenterCrop(128),
        transforms.RandomCrop(128, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Get ImageNette dataset instead of CIFAR100
    path = untar_data(URLs.IMAGENETTE_160)
    train_dataset = datasets.ImageFolder(root=str(path / 'train'), transform=None)
    test_dataset = datasets.ImageFolder(root=str(path / 'val'), transform=test_transform)

    # Split into train and validation
    train_indices, val_indices = random_split(range(len(train_dataset)), [9000, 469])
    train_dataset = datasets.ImageFolder(root=str(path / 'train'), transform=None)
    train_set = Subset(train_dataset, train_indices)

    val_dataset = datasets.ImageFolder(root=str(path / 'train'), transform=None)
    val_set = Subset(val_dataset, val_indices)

    # Add label noise to training set
    # Add label noise
    print(f"Adding {noise_level * 100}% {noise_type} label noise to training set")
    original_labels = np.array([train_set[i][1] for i in range(len(train_set))])
    noisy_labels = add_noise_to_labels(original_labels, noise_level, num_classes=10, noise_type=noise_type)

    noisy_train_set = NoisyDataset(train_set, noisy_labels, original_labels, train_transform)
    val_set.dataset.transform = test_transform

    # Define data loaders
    train_loader = DataLoader(noisy_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Set result folder
    result_folder = args.result_dir
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print("Result_folder created: ", result_folder)
    with open(result_folder + 'args.txt', 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')

    # Model - adjust for ImageNette (10 classes)
    model1 = create_model()
    model2 = create_model() if method == 'coteaching' else None
    # Loss
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
    # Optimizers
    optimizer1 = optim.AdamW(model1.parameters(), lr=1e-3, weight_decay=5e-4)
    optimizer2 = optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=5e-4) if method == 'coteaching' else None
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=max_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=max_epochs) if method == 'coteaching' else None

    # Co-teaching handler
    coteach = CoTeaching(model1, model2, optimizer1, optimizer2, criterion,
                         forget_rate=noise_level) if method == 'coteaching' else None

    # Metrics
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'test_acc': 0,
        'noisy_train_acc': [],
        'clean_train_acc': []
    }

    best_acc = -1
    best_epoch = -1

    # Training loop
    for epoch in range(max_epochs):
        print("-" * 10)
        model1.train()
        if model2:
            model2.train()

        epoch_loss = 0
        clean_correct = 0
        noisy_correct = 0
        total = 0

        for inputs, noisy_labels, clean_labels in train_loader:
            inputs, noisy_labels, clean_labels = inputs.to(device), noisy_labels.to(device), clean_labels.to(device)

            if method == 'coteaching':
                loss1, loss2 = coteach.update(inputs, noisy_labels)
                epoch_loss += (loss1 + loss2) / 2
            else:
                optimizer1.zero_grad()
                outputs = model1(inputs)
                loss = criterion(outputs, noisy_labels).mean()
                loss.backward()
                optimizer1.step()
                epoch_loss += loss.item()

            # Track accuracy on clean and noisy labels
            with torch.no_grad():
                outputs = model1(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += clean_labels.size(0)
                clean_correct += (predicted == clean_labels).sum().item()
                noisy_correct += (predicted == noisy_labels).sum().item()

        # Update learning rate
        scheduler1.step()
        if scheduler2:
            scheduler2.step()

        # Calculate metrics
        epoch_loss /= len(train_loader)
        clean_acc = 100 * clean_correct / total
        noisy_acc = 100 * noisy_correct / total

        metrics['train_loss'].append(epoch_loss)
        metrics['clean_train_acc'].append(clean_acc)
        metrics['noisy_train_acc'].append(noisy_acc)

        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Train Accuracy (clean labels): {clean_acc:.2f}%")
        print(f"Train Accuracy (noisy labels): {noisy_acc:.2f}%")

        # Validation
        model1.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model1(inputs)
                val_loss += criterion(outputs, labels).mean()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        metrics['val_loss'].append(val_loss.cpu().numpy())
        metrics['val_acc'].append(val_acc)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model1.state_dict(), os.path.join(result_dir, "best_model.pth"))
            print("Saved new best model")

    # Test evaluation
    model1.load_state_dict(torch.load(os.path.join(result_dir, "best_model.pth")))
    model1.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model1(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    metrics['test_acc'] = test_acc

    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save metrics
    np.save(os.path.join(result_dir, 'metrics.npy'), metrics)

    # Plot results
    plt.clf()
    plt.plot(range(max_epochs), metrics['train_loss'], label='Train Loss')
    plt.plot(range(max_epochs), metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    plt.savefig(os.path.join(result_folder, "loss_history.png"))
