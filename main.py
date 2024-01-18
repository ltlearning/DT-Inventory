import csv
import time,os
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_inv import GPT, GPTConfig
from mingpt.trainer_inv import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
from uti import StateActionReturnDataset,setup_logger
from Rgenerate import generate_data

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=5000)#00
parser.add_argument('--num_buffers', type=int, default=1)
parser.add_argument('--game', type=str, default='inv')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')

parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_embd', type=int, default=128)
parser.add_argument('--vocab_size', type=int, default=17)

parser.add_argument('--LT_min', type=int, default=3)
parser.add_argument('--LT_max', type=int, default=8)
parser.add_argument('--Demand_Max', type=int, default=20)
parser.add_argument('--mu', type=int, default=8)
parser.add_argument('--h', type=int, default=1)
parser.add_argument('--p', type=int, default=5)
parser.add_argument('--max_episode_length', type=int, default=90)

args = parser.parse_args()
set_seed(args.seed)
# set up logging
name = 'DT' + time.strftime('%Y-%m-%d-%H-%M-%S')
stream = os.path.join('DTlog', name)
logger = setup_logger(name='Train', level=20, stream = stream)

logger.info(args)

generate_data(args)
bs_list=np.arange(1,17)
cap_list=np.arange(7,17)
#bs_list=[]#don't augment
cap_list=[]
logger.info("bs list: {}, cap list: {}".format(bs_list,cap_list))

obss, actions, returns, done_idxs, rtgs, timesteps, demand_validation, obss_validation = create_dataset(args, stream,args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer,90,bs_list,cap_list)#max_episode_length=90,rtgs是负的
train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

logger.info("min historical action: {}, max historical action: {}".format(np.min(actions.ravel()),np.max(actions).ravel()))
if train_dataset.vocab_size>args.vocab_size:
    raise Exception('vocab_size in train_dataset {} is larger than given vocab_size in args {}'.format(train_dataset.vocab_size,args.vocab_size))
mconf = GPTConfig(args.vocab_size, train_dataset.block_size,######train_dataset会影响最后的，block_size=args.context_length*3，
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, model_type=args.model_type, max_timestep=max(timesteps))

mconf.LT = args.LT_max#GPT.state_encoder = nn.Sequential(nn.Linear(config.LT+1, config.n_embd), nn.Tanh()),state_dimension

model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tgt_list = -np.linspace(600, 2400, num=10)
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,#epochs是测试的轮数
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps),tgt=tgt_list)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train(args, stream, logger, demand_validation, obss_validation)
