import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.ticker as ticker
from config import get_config, get_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from dataset import causal_mask
from pathlib import Path

# --- 1. PLOT THE REAL LOSS CURVE ---
def plot_real_loss_curve():
    print("--- GENERATING PLOT #7: REAL LOSS CURVE ---")
    
    # THESE ARE YOUR EXACT NUMBERS FROM THE TERMINAL
    loss_data = [
        3.049, 3.540, 2.680, 3.036, 2.145, 
        2.452, 2.721, 2.118, 2.124, 2.172, 
        1.943, 1.743, 1.972, 1.694, 1.640, 
        1.524, 1.590, 1.730, 1.547, 1.483
    ]
    epochs = range(1, len(loss_data) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_data, marker='o', linestyle='-', color='b', linewidth=2, label='Training Loss')
    
    # Add a smoothed trend line (dashed red) to look like a research paper
    z = np.polyfit(epochs, loss_data, 3)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), "r--", label="Trend Line")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Plot #7: Training Convergence (Real Data)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/final_plot_7_loss_curve.png")
    print("✅ Saved Real Loss Curve")

# --- 2. PLOT THE PERFECT SQUARE HEATMAP ---
def plot_square_heatmap():
    print("--- GENERATING PLOT #10: SQUARE HEATMAP ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    
    # Load Epoch 19
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], d_model=config['d_model']).to(device)
    
    model_filename = get_weights_file_path(config, "19")
    print(f"Loading: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # TRICK: Use a sentence that translates 1-to-1 to get a Square
    # "The cat sits." -> "Die Katze sitzt." (3 words -> 3 words)
    sentence = "The cat sits."

    # Encode
    enc_input_tokens = tokenizer_src.encode(sentence).ids
    sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64).to(device)
    eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64).to(device)
    pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64).to(device)

    enc_tensor = torch.tensor(enc_input_tokens, dtype=torch.int64).to(device)
    encoder_input = torch.cat([sos_token, enc_tensor, eos_token], dim=0)
    
    # Pad remainder
    pad_len = config['seq_len'] - len(encoder_input)
    encoder_input_padded = torch.cat([encoder_input, torch.tensor([pad_token] * pad_len).to(device)], dim=0)
    
    source_mask = (encoder_input_padded != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
    encoder_input_padded = encoder_input_padded.unsqueeze(0)

    # Decode
    decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).long().to(device)
    while True:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(model.encode(encoder_input_padded, source_mask), source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input_padded).fill_(next_word.item()).to(device)], dim=1)
        if next_word == tokenizer_tgt.token_to_id('[EOS]'): break

    # Get Attention
    attn = model.decoder.layers[-1].cross_attention_block.attention_scores[0].cpu().detach().numpy()
    attn_avg = attn.mean(axis=0) # Average Heads

    # CROP THE MATRIX (Remove Padding, Remove SOS/EOS offset)
    # This forces the visual to focus only on the core words
    len_src = len(enc_input_tokens) # 3 words
    len_tgt = len(decoder_input[0]) - 2 # Exclude SOS and EOS
    
    # Slice the matrix
    # Y-axis: Generated Words
    # X-axis: Source Words
    visual_matrix = attn_avg[1:len_tgt+1, 1:len_src+1]

    fig = plt.figure(figsize=(6, 6)) # FORCE SQUARE FIGURE
    ax = fig.add_subplot(111)
    cax = ax.matshow(visual_matrix, cmap='viridis')
    fig.colorbar(cax)

    # Labels
    x_labels = [tokenizer_src.id_to_token(i) for i in enc_input_tokens]
    y_labels = [tokenizer_tgt.id_to_token(i) for i in decoder_input[0][1:-1].cpu().numpy()]

    ax.set_xticklabels([''] + x_labels, rotation=90, fontsize=12)
    ax.set_yticklabels([''] + y_labels, fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.title("Plot #10: Aligned Cross-Attention", fontsize=14)
    plt.savefig("results/final_plot_10_heatmap.png")
    print("✅ Saved Square Heatmap")

if __name__ == "__main__":
    try:
        plot_real_loss_curve()
        plot_square_heatmap()
    except Exception as e:
        print(f"Error: {e}")