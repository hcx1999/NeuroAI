# CIFAR-10 Vanilla SNN

Part I, task 1: Train a baseline spiking convolutional network on CIFAR-10 using SpikingJelly surrogate gradients.

## Files Structure
- `train_cifar10_snn.py`: End-to-end script for data loading, training, evaluation, and checkpointing.

## Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run with default hyperparameters (CPU/GPU auto-detected):
```bash
python train_cifar10_snn.py --save-dir runs
```
Check available options:
```bash
python train_cifar10_snn.py --help
```
