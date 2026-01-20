import torch
import torchmetrics
from config import get_config
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from dataset import BilingualDataset, causal_mask
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizers(config, epoch_to_load):
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], d_model=config['d_model']).to(device)
    
    # Load Epoch 19 (The Smart One)
    model_filename = f"{config['model_folder']}/{config['model_basename']}{epoch_to_load:02d}.pt"
    print(f"Loading weights from: {model_filename}")
    
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer_src, tokenizer_tgt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len: break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(model.encode(source, source_mask), source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word == eos_idx: break
    return decoder_input.squeeze(0)

def calculate_bleu(config, epoch_num):
    print(f"--- CALCULATING BLEU SCORE (Epoch {epoch_num}) ---")
    model, tokenizer_src, tokenizer_tgt = load_model_and_tokenizers(config, epoch_num)
    model.eval()
    
    # Use 100 test samples
    ds = load_dataset(config['datasource'], split='test[:100]')
    
    predicted = []
    expected = []
    
    print("Translating test sentences...")
    for item in tqdm(ds):
        src = item['en']
        tgt = item['de']
        
        enc_input_tokens = tokenizer_src.encode(src).ids
        enc_input = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config['seq_len'] - len(enc_input_tokens) - 2), dtype=torch.int64)
        ], dim=0).to(device)
        src_mask = (enc_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
        
        model_out = greedy_decode(model, enc_input.unsqueeze(0), src_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        model_txt = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
        
        predicted.append(model_txt)
        expected.append([tgt])
        
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    print(f"\n✅ FINAL BLEU SCORE: {bleu.item() * 100:.2f}")
    if bleu.item() * 100 > 25:
        print("🏆 RESULT: RESEARCH GRADE QUALITY ACHIEVED!")

def visualize_encoder_self_attention(config, epoch_num):
    print(f"\n--- GENERATING PLOT #8: Encoder Self-Attention (Epoch {epoch_num}) ---")
    model, tokenizer_src, tokenizer_tgt = load_model_and_tokenizers(config, epoch_num)
    model.eval()
    
    sentence = "A dog runs in the snow."
    enc_input_tokens = tokenizer_src.encode(sentence).ids
    enc_input = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64),
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config['seq_len'] - len(enc_input_tokens) - 2), dtype=torch.int64)
    ], dim=0).to(device)
    src_mask = (enc_input != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
    
    model.encode(enc_input.unsqueeze(0), src_mask)
    attn = model.encoder.layers[0].self_attention_block.attention_scores[0, 0].cpu().detach().numpy()
    
    plt.figure(figsize=(8,8))
    limit = len(enc_input_tokens) + 2
    plt.matshow(attn[:limit, :limit], cmap='bone', fignum=1)
    
    token_labels = ['[SOS]'] + [tokenizer_src.id_to_token(i) for i in enc_input_tokens] + ['[EOS]']
    plt.xticks(range(len(token_labels)), token_labels, rotation=90)
    plt.yticks(range(len(token_labels)), token_labels)
    
    plt.title(f"Plot #8: Encoder Self-Attention (Epoch {epoch_num})")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/plot_8_encoder_self_attn.png")
    print("✅ Plot #8 saved.")

if __name__ == "__main__":
    config = get_config()
    EPOCH = 19  # <--- THE SMART BRAIN
    
    calculate_bleu(config, EPOCH)
    visualize_encoder_self_attention(config, EPOCH)