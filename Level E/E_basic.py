

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils as utils

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CNN(in_channels=3, num_classes=10).to(device)


    # test_model
    # x = torch.randn(2, 3, 32, 32).to(device)
    # print(model(x).shape)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),  weight_decay=1e-3)  # L2 regularization # weight decay  # adjust
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=3e-3, step_size_up=3900, mode='triangular2', cycle_momentum=False)  #adjust

    train_progress = TrainProgress(batch_size=32, epoch=30,  criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, model=model)
    train_loss_list, train_accuracy_list = train_progress.train()
    train_progress.result()
    # Plot training loss and accuracy
    train_progress.plot_metrics(train_loss_list, train_accuracy_list)


class TrainProgress:
    def __init__(self, batch_size, epoch, criterion, optimizer, scheduler, device, model):
        # cifar 10
        self.classes = ('plan', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.K = 10
        self.d = 32*32*3
        self.batch_size = batch_size
        self.epoch = epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model = model


    @staticmethod
    def data_cifar():
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
             transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
             transforms.Normalize(
                 mean=(0.5, 0.5, 0.5),  # 3 dimension  -mean / st
                 std=(0.5, 0.5, 0.5))  #
             ])

        transform1 = transforms.Compose(
            [transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
             transforms.Normalize(
                 mean=(0.5, 0.5, 0.5),  # 3 dimension  -mean / st
                 std=(0.5, 0.5, 0.5))  #
             ])

        train_set = datasets.CIFAR10(
            root="./try_dataset",  # download file
            train=True,  # download training dataset
            download=True,  # download
            transform=transform  # range [0, 255] -> [0.0,1.0])
            )

        train_load = DataLoader(
            train_set,
            batch_size=100,
            shuffle=True,
            num_workers=2
        )

        test_set = datasets.CIFAR10(
            root="./try_dataset",  # download file
            train=False,  # download testing dataset
            download=True,  # download
            transform=transform1  # range [0, 255] -> [0.0,1.0])
        )

        test_load = DataLoader(
            test_set,
            batch_size=100,
            shuffle=False,
            num_workers=2
        )

        classes = ('plan', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return train_load, test_load

    def train(self):
        # Load Data
        # train_loader, val_loader, test_loader = self.delt_with_data()
        train_loader, _ = self.data_cifar()

        # Tracking metrics
        train_loss = []
        train_accuracy = []

        for epoch in range(self.epoch):
            self.model.train()
            running_loss = 0
            train_correct = 0
            total_samples = 0

            for inputs, labels in train_loader:
                # push to GPU device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()

                utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Scheduler step - overwrite the optimizer lr
                self.scheduler.step()

                # Track running loss
                running_loss += loss.item() * labels.size(0)

                # Track accuracy
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / total_samples
            epoch_acc = train_correct / total_samples

            # Save for plotting
            train_loss.append(epoch_loss)
            train_accuracy.append(epoch_acc)
            # Print progress
            print(f"Epoch {epoch + 1}/{self.epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}")

        print("Training complete.")
        return train_loss,  train_accuracy

    def result(self):
        _, test_loader = self.data_cifar()
        self.model.eval()
        test_correct = 0
        total = 0
        test_loss = 0

        # if it would like to detect each classes correctness separately
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]


        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # if it would like to detect each classes correctness separately
                for i in range(self.batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            test_loss = test_loss / total
            test_accuracy = test_correct / total
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            # if it would like to detect each classes correctness separately
            for i in range(10):
                acc = 100 * n_class_correct[i]/n_class_samples[i]
                print(f'Accuracy of {self.classes[i]}: {acc} %')

        return test_loss, test_accuracy

    @staticmethod
    def plot_metrics(losses, accuracies):
        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Training Loss")
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('fig/Train and Valid Loss.png')
        plt.show()

        # Plot Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(accuracies, label="Training Accuracy")
        plt.title("Training Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.savefig('fig/Train and Valid Accuracy.png')
        plt.legend()
        plt.show()



VGG = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']  # adjust
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.patchify = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=64,  kernel_size=(2, 2), stride=(2, 2)),  # how to write patchify layer?
                                      nn.BatchNorm2d(64),  #adjust
                                      nn.ReLU()
                                      )

        self.conv_layers = self.create_conv_layer(VGG, in_channels=64)

        self.fcs = nn.Sequential(
            nn.Linear(256*2*2, 128),  # adjust
            nn.ReLU(),
            nn.Dropout(p=0.5),  # adjust
            nn.Linear(128, num_classes),
            )
    def forward(self, x):
        x = self.patchify(x)
        x = self.conv_layers(x)
        # print(f"Shape before flattening: {x.shape}")
        x = x.reshape(x.shape[0], -1)  # flatten?
        # print(f"Shape after flattening: {x.shape}")
        x = self.fcs(x)
        return x

    @staticmethod
    def create_conv_layer(architecture, in_channels):
        layer = []

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layer += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                          nn.BatchNorm2d(x),
                          nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layer += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layer)





if __name__ == "__main__":
    main()
