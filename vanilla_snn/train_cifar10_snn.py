import os
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# SpikingJelly
from spikingjelly.activation_based import neuron, functional, surrogate, layer


# ---------------------
# Model: Conv SNN
# ---------------------
class ConvSNN(nn.Module):
    def __init__(self, num_classes: int = 10, T_steps: int = 4, dropout_p: float = 0.2):
        super().__init__()
        self.T_steps = T_steps
        sg = surrogate.ATan()  # activation-driven surrogate gradient
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.sn1 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.sn2 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        self.pool = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.sn3 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        # Classifier
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(dropout_p)
        self.sn_fc1 = neuron.IFNode(surrogate_function=sg, detach_reset=True)

        self.fc2 = nn.Linear(512, num_classes)
        # Last spiking layer optional; use membrane as logits via rate coding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)  # reset neuron states per sample/mini-batch
        # Accumulate spikes over time and compute rate-coded logits
        logits_acc = 0.0
        for t in range(self.T_steps):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.sn1(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.sn2(out)
            out = self.pool(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.sn3(out)
            out = self.pool(out)

            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.sn_fc1(out)

            logits = self.fc2(out)
            logits_acc = logits_acc + logits
        return logits_acc / float(self.T_steps)


# ---------------------
# Data
# ---------------------


def get_dataloaders(
    batch_size: int = 128, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    root = os.path.join(os.path.dirname(__file__), "..", "data")
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, transform=train_tf, download=True
    )
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, transform=test_tf, download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------
# Train & Eval
# ---------------------


def train_one_epoch(
    model, loader, criterion, optimizer, device, grad_clip: float, progress: bool
):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    data_iter = (
        tqdm(loader, desc="Train", leave=False, ncols=80) if progress else loader
    )
    for images, targets in data_iter:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, progress: bool):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    data_iter = tqdm(loader, desc="Eval", leave=False, ncols=80) if progress else loader
    with torch.no_grad():
        for images, targets in data_iter:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
    return running_loss / total, correct / total


# ---------------------
# Main
# ---------------------


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Conv SNN with activation-driven BP (CPU)"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--step-size", type=int, default=20, help="StepLR step size (epochs)"
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR decay factor")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Clip grad norm; <=0 to disable"
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Enable tqdm progress bar (default)",
    )
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default=os.path.join("runs"))
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = ConvSNN(num_classes=10, T_steps=args.timesteps).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.grad_clip,
            args.progress,
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, args.progress
        )
        dt = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Test loss {test_loss:.4f} acc {test_acc:.4f} | lr {scheduler.get_last_lr()[0]:.5f} | {dt:.1f}s"
        )

        scheduler.step()

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "timesteps": args.timesteps,
            }
            torch.save(
                ckpt,
                os.path.join(args.save_dir, f"best_snn_timesteps{args.timesteps}.pt"),
            )

    print(f"Best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
