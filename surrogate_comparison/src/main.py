import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, functional, layer
import time
import pandas as pd
import os
import torch.multiprocessing as mp

from modules import SuperSpike, PiecewiseLinear, SigmoidSurrogate

class CSNN(nn.Module):
    '''
    Conv -> BN -> LIF -> MaxPool -> Conv -> BN -> LIF -> MaxPool -> FC -> LIF
    '''
    def __init__(self, T: int, surrogate_function: nn.Module):
        super().__init__()
        self.T = T
        self.features = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True),
            layer.MaxPool2d(2, 2),  # 32x32 -> 16x16

            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True),
            layer.MaxPool2d(2, 2)   # 16x16 -> 8x8
        )
        
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(128 * 8 * 8, 10, bias=False),
            neuron.LIFNode(surrogate_function=surrogate_function, detach_reset=True)
        )

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out_fr = 0.0
        for t in range(self.T):
            x_t = x_seq[t]
            x_t = self.features(x_t)
            x_t = self.classifier(x_t)
            out_fr += x_t 
        out_fr = out_fr / self.T
        return out_fr

def train_worker(surrogate_name, alpha, device_id, epochs=100, batch_size=64, T=4, save_dir='../results'):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[{surrogate_name}] Starting on {device}...")

    if surrogate_name == 'SuperSpike':
        my_surrogate = SuperSpike(alpha=alpha)
    elif surrogate_name == 'PiecewiseLinear':
        my_surrogate = PiecewiseLinear(alpha=alpha)
    elif surrogate_name == 'SigmoidSurrogate':
        my_surrogate = SigmoidSurrogate(alpha=alpha)
    else:
        raise ValueError(f"Unknown surrogate: {surrogate_name}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CSNN(T=T, surrogate_function=my_surrogate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    records = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        samples = 0
        
        start_time = time.time()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            out_fr = model(img)
            loss = F.cross_entropy(out_fr, label)
            loss.backward()
            optimizer.step()
            
            functional.reset_net(model)
            
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            samples += label.numel()
        
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / samples
        avg_train_acc = train_acc / samples

        model.eval()
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device)
                label = label.to(device)
                out_fr = model(img)
                functional.reset_net(model)
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                test_samples += label.numel()
        
        avg_test_acc = test_acc / test_samples
        
        print(f"[{surrogate_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}% | Test Acc: {avg_test_acc*100:.2f}%")

        records.append({
            'Epoch': epoch + 1,
            'Surrogate': surrogate_name,
            'Alpha': alpha,
            'Train_Loss': avg_train_loss,
            'Train_Acc': avg_train_acc,
            'Test_Acc': avg_test_acc,
            'Time': epoch_time
        })
        
        df = pd.DataFrame(records)
        df.to_csv(f"{save_dir}/result_{surrogate_name}.csv", index=False)

    print(f"[{surrogate_name}] Training Finished. Saved to result_{surrogate_name}.csv")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
    torchvision.datasets.CIFAR10(root='../data', train=False, download=True)

    experiments = [
        {'name': 'SuperSpike',       'alpha': 10.0, 'gpu_id': 0},
        {'name': 'PiecewiseLinear',  'alpha': 1.0,  'gpu_id': 0},
        {'name': 'SigmoidSurrogate', 'alpha': 4.0,  'gpu_id': 0},
    ]

    processes = []
    
    print("Starting parallel training processes...")

    for exp in experiments:
        p = mp.Process(target=train_worker, args=(exp['name'], exp['alpha'], exp['gpu_id']))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All experiments completed.")