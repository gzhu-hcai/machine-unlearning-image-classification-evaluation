import torch
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_INIT_DIR = "./checkpoints"
HYPER_PARAMETERS = {}
HP_BATCH_SIZE = 128
HP_NORMAL_SIGMA = 0.2
HP_FLOAT = 0.5
HP_LEARNING_RATE = 0.01
HP_MOMENTUM = 0.9
HP_WEIGHT_DECAY = 5e-4
def default_loaders(): return {}