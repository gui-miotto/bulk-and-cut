# Randomness control:
import torch
import numpy as np
global_seed = 0
torch.manual_seed(global_seed)
rng = np.random.default_rng(seed=global_seed)

# Pytorch device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Convinience imports:
from bulkandcut.evolution import Evolution
from bulkandcut.model import BNCmodel
from bulkandcut.dataset import load_dataset
from bulkandcut.pareto import generate_pareto_animation
from bulkandcut.pareto import Benchmark

