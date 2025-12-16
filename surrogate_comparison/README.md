# SNN Surrogate Gradient Comparison

Part I, task 2: This project compares different surrogate gradient functions for Spiking Neural Networks (SNN) on the CIFAR-10 dataset.

## Files Structure
- `surrogate.py`: Definitions of custom autograd functions (SuperSpike, PiecewiseLinear, Sigmoid).
- `modules.py`: PyTorch `nn.Module` wrappers for the surrogate functions.
- `main.py`: Main training loop (Multiprocessing supported).
- `train.py`: SNN structure, download datasets.
- `plot.py`: Visualizes the training results from CSV files.

## Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
