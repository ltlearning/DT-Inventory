import csv
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
from augment import bs_cap_augment,bs_augment
from benchmarks import test_bs,test_bs_cap

def create_dataset(args,stream,num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer, max_episode_length,bs_list,cap_list):
    # -- load data from memory (make more efficient)
    actions = np.load('lost_sales/action.npy')
    stepwise_returns = np.load('lost_sales/cost.npy')
    obss = np.load('lost_sales/state.npy')
    #For validation
    demand = np.load('lost_sales/demand.npy')
    obss_validation = obss[-10:]#(100,90,LT)
    demand_validation = demand[-10:]#(100,90)
    #else
    actions = actions[:-10]
    stepwise_returns = stepwise_returns[:-10]
    obss = obss[:-10]
    if len(bs_list)!=0:
        if len(cap_list)!=0:
            best_bs,best_cap = bs_cap_augment(bs_list,cap_list)#augment data and save as npy
            for bs in bs_list:
                for cap in cap_list:
                    try:
                        actions = np.concatenate((actions,np.load('lost_sales/cap/action_bs'+str(bs)+'cap'+str(cap)+'.npy')))
                        stepwise_returns = np.concatenate((stepwise_returns,np.load('lost_sales/cap/cost_bs'+str(bs)+'cap'+str(cap)+'.npy')))
                        obss = np.concatenate((obss,np.load('lost_sales/cap/state_bs'+str(bs)+'cap'+str(cap)+'.npy')))
                    except:
                        pass
            test_bs_cap(args, stream, [best_bs], [best_cap])
        else:
            best_bs = bs_augment(bs_list)
            for bs in bs_list:
                try:
                    obss = np.concatenate((obss,np.load('lost_sales/state_bs'+str(bs)+'.npy')))
                    actions = np.concatenate((actions,np.load('lost_sales/action_bs'+str(bs)+'.npy')))
                    stepwise_returns = np.concatenate((stepwise_returns,np.load('lost_sales/cost_bs'+str(bs)+'.npy')))
                except:
                    pass
            test_bs(args, stream, [best_bs])
    else:
        pass
        #actions = np.concatenate((actions,np.load('lost_sales/cap/1good_action_bscap.npy')))
        #stepwise_returns = np.concatenate((stepwise_returns,np.load('lost_sales/cap/1good_cost_bscap.npy')))
        #obss = np.concatenate((obss,np.load('lost_sales/cap/1good_state_bscap.npy')))
    
    returns = -np.array(stepwise_returns.sum(1))
    returns = np.append(returns,0)
    done_idxs = max_episode_length-1+(max_episode_length-1)*np.arange(actions.shape[0])
    actions = actions.ravel()
    stepwise_returns = -stepwise_returns.ravel()
    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    
    #######这是用ESPER里面的方法对rtg进行一番操作，详见simple/ESPER/return_transforms-main/return_transforms/generate.py
    #目前就不考虑了
    '''
    import pickle
    rtg_f = open('ESPER/return_transforms-main/data/inv.ret','rb')
    rtg_ = np.array(pickle.load(rtg_f))
    rtg = rtg_.ravel()*1000'''
    ################################################################################################################
    
    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))
    
    return obss.reshape(len(actions),-1)/100, actions, returns, done_idxs, rtg, timesteps, demand_validation, obss_validation#state除了100，后面测试也要除
