import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from model import build_transformer
from dataset import causal_mask
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(config, epoch_to_load):
    # Load Tokenizers
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    
    # Build Model
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], d_model=config['d_model']).to(device)
    
    # Load Weights
    model_filename = get_weights_file_path(config, f"{epoch_to_load:02d}")
    print(f"Loading weights from: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer_src, tokenizer_tgt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate output
        out = model.decode(model.encode(source, source_mask), source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def visualize_attention(model, encoder_input, decoder_input):
    # We need to run the model one last time to capture the attention weights
    # The weights are stored inside the model layers as 'self.attention_scores'
    
    # 1. Get Cross Attention from the last Decoder Layer
    # Structure: model -> decoder -> layers[-1] -> cross_attention_block -> attention_scores
    attn = model.decoder.layers[-1].cross_attention_block.attention_scores[0].cpu().detach().numpy()
    # Shape: (Heads, Seq_Len_Decoder, Seq_Len_Encoder)
    
    # Just take the first head for visualization
    attn_head = attn[0] 
    return attn_head

def run_translation(sentence: str, epoch: int):
    config = get_config()
    model, tokenizer_src, tokenizer_tgt = load_trained_model(config, epoch)
    model.eval()

    # Encode the source sentence
    sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    encoder_input_tokens = tokenizer_src.encode(sentence).ids
    encoder_input = torch.cat(
        [sos_token, torch.tensor(encoder_input_tokens, dtype=torch.int64), eos_token, torch.tensor([pad_token] * (config['seq_len'] - len(encoder_input_tokens) - 2), dtype=torch.int64)],
        dim=0
    ).to(device)
    
    source_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
    encoder_input = encoder_input.unsqueeze(0) # Batch size 1

    # Run Inference
    model_out = greedy_decode(model, encoder_input, source_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
    
    # Convert back to text
    source_text = sentence
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
    print(f"\nSOURCE: {source_text}")
    print(f"PREDICTED: {model_out_text}")
    
    # PLOT HEATMAP
    attn_matrix = visualize_attention(model, encoder_input, model_out)
    
    # Filter matrix to remove padding (makes the plot look nicer)
    len_src = len(encoder_input_tokens) + 2
    len_tgt = len(model_out)
    attn_matrix = attn_matrix[:len_tgt, :len_src]
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_matrix, cmap='bone')
    fig.colorbar(cax)

    # Set labels
    ax.set_xticklabels([''] + [tokenizer_src.id_to_token(i) for i in encoder_input.squeeze(0)[:len_src].cpu().numpy()], rotation=90)
    ax.set_yticklabels([''] + [tokenizer_tgt.id_to_token(i) for i in model_out.cpu().numpy()])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(f"Plot #10: Cross-Attention Alignment (Epoch {epoch})")
    
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/plot_10_cross_attention.png")
    print("✅ Heatmap saved to results/plot_10_cross_attention.png")

if __name__ == "__main__":
    # epoch to load (usually 0 if you stopped early)
    EPOCH_TO_TEST = 0 
    
    print("\n" + "="*50)
    print("🤖 INTERACTIVE TRANSLATOR (English -> German)")
    print("="*50)

    while True:
        user_sentence = input("\nEnter English Sentence (or 'q' to quit): ")
        
        if user_sentence.lower() == 'q':
            break
            
        try:
            run_translation(user_sentence, EPOCH_TO_TEST)
        except FileNotFoundError:
            print("❌ Error: Model weight file not found. Did you train the model?")
        except Exception as e:
            print(f"❌ Error: {e}")