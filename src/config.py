from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 10,       # lighter
        "lr": 10**-4,
        "seq_len": 150,         # lighter
        "d_model": 256,         # lighter
        "d_ff": 512,            # lighter
        "N": 3,                 # lighter
        "h": 4,                 # lighter
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