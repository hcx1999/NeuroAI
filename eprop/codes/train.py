import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
from time import time

def load_pixel_by_pixel_data(file_path, batch_size=64, shuffle=True, normalize=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from {file_path}...", flush=True)
    data_dict = torch.load(file_path, map_location='cpu')
    if normalize:
        data = data_dict['data'].float() / 255.0
    else:
        data = data_dict['data'].float()
    
    if data.dim() == 2:
        data = data.unsqueeze(-1)
        
    targets = data_dict['targets'].long()
    dataset = TensorDataset(data, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), data.shape[-1]

def train_epoch(model, loader, optimizer, criterion, device, max_norm=1.0):
    start_time = time()
    model.train()
    total_loss, correct, total_samples = 0, 0, 0

    if hasattr(model, 'model'):
        if hasattr(model.model, 'w_in'):
            for p in model.model.w_in.parameters():
                p.requires_grad_(False)
        if hasattr(model.model, 'w_rec'):
            for p in model.model.w_rec.parameters():
                p.requires_grad_(False)
        if hasattr(model.model, 'w_out'):
            for p in model.model.w_out.parameters():
                p.requires_grad_(True)

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)

        optimizer.zero_grad()
        final_readout_seq, traces = model(inputs)

        final_readout = final_readout_seq[:, -1, :]
        loss = criterion(final_readout, labels)

        with torch.no_grad():
            y_prob = F.softmax(final_readout, dim=1)
            y_target = F.one_hot(labels, num_classes=final_readout.shape[1]).float().to(device)
            delta_out = y_prob - y_target

            B = model.get_broadcast_matrix()
            learning_signal = torch.matmul(delta_out, B)

            f_e_in_T = traces[0]
            f_e_rec_T = traces[1]

            grad_w_in = torch.einsum('bh,bhi->hi', learning_signal, f_e_in_T) / batch_size
            grad_w_rec = torch.einsum('bh,bhj->hj', learning_signal, f_e_rec_T) / batch_size
            
            if hasattr(model.model, 'w_in'):
                model.model.w_in.weight.grad = grad_w_in
            if hasattr(model.model, 'w_rec'):
                model.model.w_rec.weight.grad = grad_w_rec

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        total_loss += loss.item()
        pred = final_readout.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total_samples += labels.size(0)

    print(f"Epoch Time: {time() - start_time:.2f}s", flush=True)
    return total_loss / len(loader), 100 * correct / total_samples

def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            final_readout_seq, _ = model(inputs)
            final_readout = final_readout_seq[:, -1, :]
            
            loss = criterion(final_readout, labels)
            
            total_loss += loss.item()
            pred = final_readout.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total_samples += labels.size(0)
            
    return total_loss / len(loader), 100 * correct / total_samples