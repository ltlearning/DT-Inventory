import time,os
import sys
import logging
import torch
import numpy as np
_streams = {
    "stdout": sys.stdout
}
from torch.utils.data import Dataset

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = int(max(actions) + 1)
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)#长度不超过block_size，保证在同一个epoch里面
                break
        idx = done_idx - block_size#长度不超过block_size，保证在同一个epoch里面
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)#只有一位？
        return states, actions, rtgs, timesteps
        
def setup_logger(name: str, level: int, stream: str = "stdout") -> logging.Logger:
    global _streams
    if stream not in _streams:
        log_folder = os.path.dirname(stream)
        _streams[stream] = open(stream, 'w')
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    for stream in _streams:
        sh = logging.StreamHandler(stream=_streams[stream])
        sh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger