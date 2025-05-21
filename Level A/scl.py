import copy
import os
import random
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision.models import resnet18, resnet50
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        img = img.clone()
        h, w = img.shape[1], img.shape[2]

        for _ in range(self.n_holes):
            y = random.randint(0, h - self.length)
            x = random.randint(0, w - self.length)

            img[:, y:y + self.length, x:x + self.length] = 0
        return img


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # Contrastive mask
            mask = torch.eq(labels, labels.T).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)
            # Exclude itself
            self_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
                0
            )
            mask = mask * self_mask
        else:
            mask = torch.eye(batch_size * contrast_count, dtype=torch.float32).to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # Calculate similarity
        similarity = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # Stabalize
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Calculate prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # Loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TwoViewDataset:
    def __init__(self, dataset, transform, transform_contrast):
        """
        Contrastive data pair

        :param dataset: Training set
        :param transform: Training transform
        :param transform_contrast: Contrastive transform
        """
        self.dataset = dataset
        self.train_transform = transform
        self.train_transform_contrast = transform_contrast

    def __getitem__(self, index):
        img, label = self.dataset[index]

        view1 = self.train_transform(img)
        view2 = self.train_transform_contrast(img)
        return (view1, view2), label

    def __len__(self):
        return len(self.dataset)


class ResNetWithProjection(nn.Module):
    def __init__(self, base_model='resnet18', feat_dim=128):
        super(ResNetWithProjection, self).__init__()
        if base_model == 'resnet18':
            self.encoder = resnet18(weights=None)
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.maxpool = nn.Identity()
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError('Unsupported base model')

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim, bias=False)
        )

    def forward(self, x):
        features = self.encoder(x)
        projection = self.projector(features)
        return F.normalize(projection, dim=1)


# Simple MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=100):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)



if __name__ == '__main__':
    # Set args
    parser = argparse.ArgumentParser('D_level', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    # Dataset
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # Train
    parser.add_argument('--max_epochs', default=100, type=int)
    # Contrastive learning specific
    parser.add_argument('--feat_dim', default=128, type=int, help='Feature dimension for contrastive learning')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for contrastive loss')
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
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_transform_contrast = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.1),
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

    val_set.dataset.transform = test_transform

    contrast_train_set = TwoViewDataset(copy.deepcopy(train_set), train_transform, train_transform_contrast)
    train_set.dataset.transform = train_transform

    # Define data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    contrast_train_loader = DataLoader(contrast_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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

    # Contrastive pre-train
    print("Starting contrastive pre-training...")
    contrastive_model = ResNetWithProjection(base_model='resnet18', feat_dim=args.feat_dim).to(device)
    contrastive_criterion = SupConLoss(temperature=args.temperature).to(device)
    contrastive_optimizer = optim.AdamW(contrastive_model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_scheduler = LinearLR(contrastive_optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(contrastive_optimizer, T_max=max_epochs - 5)
    contrastive_scheduler = SequentialLR(contrastive_optimizer,
                                         [warmup_scheduler, cosine_scheduler],
                                         milestones=[5])

    # Train contrastive model
    for epoch in range(max_epochs):
        print("-" * 10)
        contrastive_model.train()
        epoch_loss = 0
        step = 0
        for idx, batch_data in enumerate(contrast_train_loader):
            step += 1
            (view1, view2), labels = batch_data
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

            contrastive_optimizer.zero_grad()

            # Get two features
            features1 = contrastive_model(view1)
            features2 = contrastive_model(view2)

            # Combine features
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

            loss = contrastive_criterion(features, labels)
            loss.backward()
            contrastive_optimizer.step()

            epoch_loss += loss.item()
            print(f"Contrastive pre-train epoch {epoch + 1}/{max_epochs}",
                  f"lr: {contrastive_optimizer.param_groups[0]['lr']:.4e}",
                  f"{step}/{len(contrast_train_set) // contrast_train_loader.batch_size}, "
                  f"train_loss: {loss.item():.4f}")

        contrastive_scheduler.step()
        epoch_loss /= step
        print(f"Contrastive epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Save model of first stages
    torch.save(contrastive_model.state_dict(), os.path.join(result_folder, "contrastive_model.pth"))
    print("Contrastive pre-training completed and model saved.")

    # Train classifier
    print("Starting MLP classifier training...")

    for param in contrastive_model.parameters():
        param.requires_grad = False

    classifier = MLPClassifier(input_dim=512, num_classes=100).to(device)

    feature_extractor = contrastive_model.encoder

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epochs)

    loss_history_train = []
    loss_history_val = []
    acc_history_val = []
    best_acc = -1
    best_metric_epoch = -1
    best_params = None

    for epoch in range(max_epochs):
        print("-" * 10)
        classifier.train()
        feature_extractor.eval()
        epoch_loss = 0
        step = 0
        for idx, batch_data in enumerate(train_loader):
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()

            # Get features
            with torch.no_grad():
                features = feature_extractor(inputs)

            outputs = classifier(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(f"Classifier epoch {epoch + 1}/{max_epochs}",
                  f"lr: {optimizer.param_groups[0]['lr']:.4e}",
                  f"{step}/{len(train_set) // train_loader.batch_size}, "
                  f"train_loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss /= step
        loss_history_train.append(epoch_loss)
        print(f"Classifier epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation
        with torch.no_grad():
            classifier.eval()
            feature_extractor.eval()
            group_pred = []
            group_label = []
            loss_val = 0
            step = 0

            for idx, val_data in enumerate(val_loader):
                step += 1
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

                features = feature_extractor(val_images)
                output = classifier(features)
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
            best_params = classifier.state_dict()
            torch.save(classifier.state_dict(), os.path.join(result_folder, "best_classifier.pth"))
            print("Saved new best classifier model")

    # Test
    print("Testing...")
    classifier.load_state_dict(best_params)
    classifier.eval()
    feature_extractor.eval()

    loss_test = 0
    step = 0
    group_pred = []
    group_label = []

    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            step += 1
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)

            features = feature_extractor(test_images)
            output = classifier(features)
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
        f"Test Loss: {loss_test:.4f}"
    )

    plt.clf()
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(max_epochs), loss_history_train, label="Train Loss")
    plt.plot(range(max_epochs), loss_history_val, label="Val Loss")
    plt.legend()
    plt.savefig(os.path.join(result_folder, "loss_history.png"))

    plt.clf()
    plt.title("Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(max_epochs), acc_history_val, label="Val Accuracy")
    plt.savefig(os.path.join(result_folder, "acc_history.png"))
