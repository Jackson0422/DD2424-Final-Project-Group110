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

from timm.layers import trunc_normal_, DropPath
from timm.models import register_model

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)  # (N, C, H, W)
        x = x.permute(0, 2, 3, 1)  # to (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtCIFAR100(nn.Module):
    def __init__(self, in_chans=3, num_classes=100,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1.):
        super().__init__()

        # Stem: smaller kernel and stride to fit CIFAR-100 input size 32x32
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # Global average pooling
        x = x.mean([-2, -1])  # (N, C)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
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

    # # Model - baseline resnet18
    # model = resnet18(weights=None)
    # # # model = resnet50(weights=None)
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # model.maxpool = nn.Identity()
    #
    # model.fc = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(model.fc.in_features, 100)
    # )

    ## Model - ConvNeXt
    # modify num_classes based on the import dataset of CIFAR10 or CIFAR100
    model = ConvNeXtCIFAR100(drop_path_rate=0.15,num_classes=100)

    model = model.to(device)
    

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
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
    plt.savefig(os.path.join(result_folder, "loss_history_conv.png"))

    plt.clf()
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(max_epochs), acc_history_val, label="Val Accuracy")
    plt.savefig(os.path.join(result_folder, "acc_history_conv.png"))
