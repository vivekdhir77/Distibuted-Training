from pathlib import Path
import os
import glob

class ModelConfig:
    def __init__(self, batch_size, num_epochs, learning_rate, model_folder, model_basename, 
                 preload, tokenizer_file, image_size, patch_size, embed_dim, num_heads, 
                 num_layers, num_classes, forward_expansion, dropout, stochastic_depth,
                 local_rank=-1, global_rank=-1):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_folder = model_folder
        self.model_basename = model_basename
        self.preload = preload
        self.tokenizer_file = tokenizer_file
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.stochastic_depth = stochastic_depth


def get_default_config() -> ModelConfig:
    return ModelConfig(
        batch_size=16,
        num_epochs=10,
        learning_rate=3e-4,
        image_size=32,
        patch_size=4,
        embed_dim=192,
        num_heads=3,
        num_layers=6,
        num_classes=10,
        forward_expansion=4,
        dropout=0.1,
        stochastic_depth=0.1,
        model_folder="weights",
        model_basename="tmodel_{0:02d}.pt",
        preload="latest",
        tokenizer_file="tokenizer_{0}.json",
    )

        


def get_weights_file_path(config: ModelConfig, epoch: int) -> str:
    """Get path to model weights file for a specific epoch"""
    os.makedirs(config.model_folder, exist_ok=True)
    model_folder = config.model_folder
    model_basename = config.model_basename
    model_filename = model_basename.format(epoch)
    return str(Path('.') / model_folder / model_filename)

def get_latest_weights_file_path(config: ModelConfig) -> str:
    """Find the latest model weights file in the model folder"""
    os.makedirs(config.model_folder, exist_ok=True)
    model_folder = config.model_folder
    
    # First check for a file named latest.pt
    latest_path = os.path.join(model_folder, 'latest.pt')
    if os.path.exists(latest_path):
        return latest_path
    
    # If not found, look for the highest epoch number
    model_files = list(Path(model_folder).glob("tmodel_*.pt"))
    if not model_files:
        return None
        
    # Sort by epoch number
    model_files = sorted(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    model_filename = model_files[-1]
    return str(model_filename)