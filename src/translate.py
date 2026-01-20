import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from model import build_transformer
from dataset import causal_mask
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(config, epoch_to_load):
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], d_model=config['d_model']).to(device)
    
    model_filename = get_weights_file_path(config, f"{epoch_to_load:02d}")
    print(f"Loading weights from: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer_src, tokenizer_tgt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    decoder_input = torch.empty(1, 1).fill_(sos_idx).long().to(device)
    
    while True:
        if decoder_input.size(1) == max_len: break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(model.encode(source, source_mask), source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word_tensor = torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)
        if next_word == eos_idx: break
    return decoder_input.squeeze(0)

# --- PLOT 1: ENG -> GER (Alignment) ---
def plot_cross_attention(model, encoder_input, model_out, tokenizer_src, tokenizer_tgt, safe_name, timestamp):
    attn = model.decoder.layers[-1].cross_attention_block.attention_scores[0].cpu().detach().numpy()
    attn_avg = attn.mean(axis=0)
    
    # --- FIX: Calculate actual length (ignore padding) ---
    src_ids = encoder_input.squeeze().cpu().numpy()
    src_labels = [tokenizer_src.id_to_token(i) for i in src_ids]
    
    # Find where the sentence ends ([EOS] or [PAD])
    try:
        real_len_src = src_labels.index('[PAD]')
    except ValueError:
        real_len_src = len(src_labels)
        
    len_tgt = len(model_out)
    
    # Slice valid area only
    visual_matrix = attn_avg[:len_tgt, :real_len_src]
    x_labels = src_labels[:real_len_src]
    y_labels = [tokenizer_tgt.id_to_token(i) for i in model_out.cpu().numpy()]

    plt.figure(figsize=(8, 8))
    # Viridis for clear alignment
    ax = sns.heatmap(visual_matrix, cmap='viridis', square=True, cbar=True,
                     xticklabels=x_labels, yticklabels=y_labels)
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Cross-Attention (Eng -> Ger)")
    plt.tight_layout()
    plt.savefig(f"results/{safe_name}_cross_attn_{timestamp}.png")
    print(f"✅ Saved Eng->Ger Heatmap")
    plt.close()

# --- PLOT 2: GER -> GER (The Triangle) ---
def plot_self_attention_triangle(model, model_out, tokenizer_tgt, safe_name, timestamp):
    attn = model.decoder.layers[0].self_attention_block.attention_scores[0].cpu().detach().numpy()
    attn_avg = attn.mean(axis=0)
    
    len_tgt = len(model_out)
    visual_matrix = attn_avg[:len_tgt, :len_tgt]
    mask = np.triu(np.ones_like(visual_matrix, dtype=bool), k=1)
    
    labels = [tokenizer_tgt.id_to_token(i) for i in model_out.cpu().numpy()]
    
    plt.figure(figsize=(8, 8))
    # RdBu_r for the Paper look
    ax = sns.heatmap(visual_matrix, mask=mask, cmap='RdBu_r', square=True, cbar=True, 
                     linewidths=0.5, linecolor='white',
                     xticklabels=labels, yticklabels=labels)
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title("Decoder Self-Attention (Causal Triangle)")
    plt.tight_layout()
    plt.savefig(f"results/{safe_name}_triangle_attn_{timestamp}.png")
    print(f"✅ Saved Triangular Heatmap")
    plt.close()

def run_translation(sentence: str, epoch: int):
    config = get_config()
    model, tokenizer_src, tokenizer_tgt = load_trained_model(config, epoch)
    model.eval()

    sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64).to(device)
    eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64).to(device)
    pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64).to(device)

    encoder_input_tokens = tokenizer_src.encode(sentence).ids
    encoder_token_tensor = torch.tensor(encoder_input_tokens, dtype=torch.int64).to(device)
    padding_tensor = torch.tensor([tokenizer_src.token_to_id("[PAD]")] * (config['seq_len'] - len(encoder_input_tokens) - 2), dtype=torch.int64).to(device)

    encoder_input = torch.cat([sos_token, encoder_token_tensor, eos_token, padding_tensor], dim=0)
    source_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
    encoder_input = encoder_input.unsqueeze(0) 

    model_out = greedy_decode(model, encoder_input, source_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
    
    source_text = sentence
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
    print(f"\nSOURCE: {source_text}")
    print(f"PREDICTED: {model_out_text}")
    
    os.makedirs("results", exist_ok=True)
    safe_name = "".join([c if c.isalnum() else "_" for c in sentence])[:15] 
    timestamp = int(time.time())
    
    plot_cross_attention(model, encoder_input, model_out, tokenizer_src, tokenizer_tgt, safe_name, timestamp)
    
    tgt_mask = causal_mask(model_out.size(0)).type_as(source_mask).to(device)
    model.decode(model.encode(encoder_input, source_mask), source_mask, model_out.unsqueeze(0), tgt_mask)
    plot_self_attention_triangle(model, model_out, tokenizer_tgt, safe_name, timestamp)

if __name__ == "__main__":
    EPOCH_TO_TEST = 19
    
    print("\n" + "="*50)
    print("🤖 INTERACTIVE TRANSLATOR (Generates Triangles!)")
    print("="*50)

    while True:
        user_sentence = input("\nEnter English Sentence (or 'q' to quit): ")
        if user_sentence.lower() == 'q': break 
        try:
            run_translation(user_sentence, EPOCH_TO_TEST)
        except Exception as e:
            print(f"❌ Error: {e}")