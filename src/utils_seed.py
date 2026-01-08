# src/utils_seed.py
import os, random
import numpy as np
import torch

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让结果更可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
