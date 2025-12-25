import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from model import RSNN
from train import load_pixel_by_pixel_data, train_epoch, test_epoch

def load_data(args):
    train_path = os.path.join(args.dataset_dir, "train.pt")
    test_path = os.path.join(args.dataset_dir, "test.pt")
    
    print("Loading Dataset...", flush=True)
    train_loader, input_dim = load_pixel_by_pixel_data(train_path, args.batch_size, shuffle=True)
    test_loader, _ = load_pixel_by_pixel_data(test_path, args.batch_size, shuffle=False)
    print(f"Input Dimension: {input_dim}", flush=True)
    
    output_size = train_loader.dataset.tensors[1].max().item() + 1    # 0~9
    return train_loader, test_loader, input_dim, output_size

def main(args):
    print(f"Using Device: {args.device} | Mode: {args.mode}", flush=True)
    train_loader, test_loader, input_dim, output_size = load_data(args)

    print(f"Initializing Model (Hidden={args.hidden_size})...")
    model = RSNN(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        output_size=output_size,
        mode=args.mode,
        broadcast=args.broadcast,
        alif_tau_a=args.alif_tau_a,
        alif_beta=args.alif_beta,
        alif_v_th_base=args.alif_v_th_base,
        alif_gamma=args.alif_gamma,
    )
    model.to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    print(f"Starting training for {args.epochs} epochs...", flush=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, args.device, args.max_norm
        )
        test_loss, test_acc = test_epoch(model, test_loader, criterion, args.device)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
            f"Test Loss: {test_loss:.4f} (Acc: {test_acc:.2f}%)",
            flush=True
        )

    if args.mode == "LIF":
        csv_filename = (
            f"log_{args.mode}_{args.broadcast}_"
            f"lr{args.lr}_hs{args.hidden_size}_"
            f"epochs{args.epochs}.csv"
        )
    else:
        csv_filename = (
            f"log_{args.mode}_{args.broadcast}_"
            f"lr{args.lr}_hs{args.hidden_size}_"
            f"taua{args.alif_tau_a}_"
            f"beta{args.alif_beta}_"
            f"vth{args.alif_v_th_base}_"
            f"gamma{args.alif_gamma}_"
            f"epochs{args.epochs}.csv"
        )
    
    csv_filename = os.path.join(args.results_dir, csv_filename)
    os.makedirs(args.results_dir, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(csv_filename, index=False)
    
    print(f"\nTraining Finished.")
    print(f"Best Test Accuracy: {df['test_acc'].max():.2f}%")
    print(f"Logs saved to: {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple RSNN Training")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="ALIF", choices=["LIF", "ALIF"])
    parser.add_argument("--broadcast", type=str, default="random", choices=["random", "symmetric"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_dir", type=str, default="/storage3/mhausserlab/neuroai/eprop/sequential_mnist/")
    parser.add_argument("--alif_tau_a", type=float, default=2000.0)
    parser.add_argument("--alif_beta", type=float, default=0.07)
    parser.add_argument("--alif_v_th_base", type=float, default=0.6)
    parser.add_argument("--alif_gamma", type=float, default=0.3)
    parser.add_argument("--results_dir", type=str, default="./results/")

    args = parser.parse_args()
    main(args) 