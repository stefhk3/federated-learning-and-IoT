import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import yaml
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from fedn.utils.helpers.helpers import get_helper
import collections
from sklearn.linear_model import Perceptron

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

def create_seed_model(exp_config):
	model = Perceptron()
	
	return model