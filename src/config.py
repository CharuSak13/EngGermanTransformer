from pathlib import Path

def get_config():
    return {
        "batch_size": 8,        # Keep small for RTX 3050 (4GB VRAM)
        "num_epochs": 20,       # Train fully
        "lr": 10**-4,
        "seq_len": 350,         # Full sentence length
        "d_model": 512,         # Paper Standard (Big Brain)
        "d_ff": 2048,           # Paper Standard
        "N": 6,                 # 6 Layers (Deep Network)
        "h": 8,                 # 8 Attention Heads
        "dropout": 0.1,
        "datasource": 'bentrevett/multi30k',
        "lang_src": "en",
        "lang_tgt": "de",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)