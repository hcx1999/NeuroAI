import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    files = {
        'result_SuperSpike.csv': 'SuperSpike (Fast Sigmoid)',
        'result_PiecewiseLinear.csv': 'Piecewise Linear (Triangular)',
        'result_SigmoidSurrogate.csv': 'Sigmoid Derivative'
    }

    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, (filename, label) in enumerate(files.items()):
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Skipping...")
            continue
            
        try:
            df = pd.read_csv(filename)
            plt.plot(
                df['Epoch'], 
                df['Test_Acc'] * 100,
                label=label, 
                linewidth=2,
                marker='o',
                markersize=4,
                color=colors[i % len(colors)]
            )
            
            final_acc = df['Test_Acc'].iloc[-1] * 100
            print(f"{label}: Final Test Acc = {final_acc:.2f}%")
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    plt.title('Comparison of Surrogate Gradients on CIFAR-10 (SNN)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('surrogate_comparison.png', dpi=300)
    print("Plot saved to 'surrogate_comparison.png'")
    plt.show()

if __name__ == '__main__':
    plot_comparison()