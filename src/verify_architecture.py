import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import os
from model import PositionalEncoding

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

def plot_positional_encoding():
    print("--- GENERATING PLOT #6: Positional Encoding Matrix ---")
    d_model = 512
    seq_len = 100
    dropout = 0.1
    
    pe = PositionalEncoding(d_model, seq_len, dropout)
    # Extract the matrix (it's saved in the buffer named 'pe')
    encoding_matrix = pe.pe.squeeze(0).numpy()
    
    plt.figure(figsize=(15, 8))
    plt.imshow(encoding_matrix, cmap='RdBu', aspect='auto')
    plt.xlabel(f"Embedding Dimension (d_model={d_model})")
    plt.ylabel(f"Sequence Position (seq_len={seq_len})")
    plt.title("Plot #6: Positional Encoding Heatmap (Paper Replication)")
    plt.colorbar()
    
    # Save to results folder
    save_path = "results/plot_6_positional_encoding.png"
    plt.savefig(save_path)
    print(f"✅ Plot saved to {save_path}")
    plt.close()

def plot_lr_schedule():
    print("\n--- GENERATING PLOT #5: Learning Rate Warmup ---")
    d_model = 512
    warmup_steps = 4000
    
    def get_lr(step):
        if step == 0: return 0
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    x = list(range(1, 20000))
    y = [get_lr(step) for step in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Plot #5: Warmup Learning Rate Schedule")
    plt.grid(True)
    
    save_path = "results/plot_5_lr_schedule.png"
    plt.savefig(save_path)
    print(f"✅ Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_positional_encoding()
    plot_lr_schedule()