"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image

from LS_env import transition_stochLT
from scipy.stats import poisson

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e2 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere 6
    final_tokens = 260e5 # (at what point we reach 10% of original LR) 9
    # checkpoint settings
    ckpt_path = None
    num_workers = 8 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, args, stream, logger, demand_validation, obss_validation):
        #demand_validation = torch.from_numpy(demand_validation)
        #obss_validation = torch.from_numpy(obss_validation)
        
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
            return losses
                    
        best_return = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for tgt in self.config.tgt:
            eval_return = self.validate(tgt, demand_validation, obss_validation)
            logger.info("tgt {}, validate cost: {}".format(tgt,eval_return))
            if eval_return<best_return:
                best_return = eval_return
                best_tgt = tgt
                torch.save(self.model,stream+'BestNet.pt')
        losses_list = []
        for epoch in range(config.max_epochs):
            losses = run_epoch('train', epoch_num=epoch)
            losses_list.append(losses)
            # -- pass in target returns
            for tgt in self.config.tgt:
                eval_return = self.validate(tgt, demand_validation, obss_validation)
                logger.info("tgt {}, validate cost: {}".format(tgt,eval_return))
                if eval_return<best_return:
                    best_return = eval_return
                    best_tgt = tgt
                    torch.save(self.model,stream+'BestNet.pt')
        np.save(stream+'losses_list.npy',np.array(losses_list).ravel())
        logger.info("best tgt: {}".format(best_tgt))
        self.test_returns(args,stream,logger,self.device,best_tgt)
        

    def validate(self, ret, demand_validation, obss_validation):
        self.model.train(False)
        reward_DT_list = []
        for k in range(len(demand_validation)):#5个horizon
            state = obss_validation[k]
            demand = demand_validation[k]
            reward_DT = 0
            previous_state = state[0]
            
            input_actions = []
            for i in range(90):
                current_demad = demand[i]
                reward = max(previous_state[0] - current_demad, 0) * 1 + min(previous_state[0] - current_demad, 0) * -5
                reward_DT += reward
                current_state = []
                current_inv = previous_state[0]
                
                #PREPARE for DT
                if i ==0:
                    input_state = torch.tensor(previous_state, dtype=torch.float32, device=self.device).div_(100).unsqueeze(0).unsqueeze(0)
                    rtgs = [ret]
                    sampled_action = sample(self.model.module, input_state/100, 1, temperature=1.0, sample=True, actions=None, 
                                            rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                                            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))
                else:
                    sampled_action = sample(self.model.module, input_state.unsqueeze(0)/100, 1, temperature=1.0, sample=True, 
                                            actions=torch.tensor(input_actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                                            rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                                            timesteps=(min(i, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
                
                current_action = sampled_action.detach().cpu().numpy()[0,-1]
                if i == 89:
                    break

                #current_state的变化跟previous_state和current_demad，current_action以及state有关
                current_state += [max(current_inv - current_demad,0) + previous_state[1]]#第一位
                for j in range(1,len(state[i+1])-1):
                    if state[i+1][j]==0:
                        current_state[0] += previous_state[j+1]
                        current_state.append(0)
                    else:
                        current_state.append(previous_state[j+1])
                current_state.append(current_action)#最后一位
                previous_state = current_state
                
                #PREPARE for DT
                input_state = torch.cat([input_state, torch.tensor(current_state).unsqueeze(0).unsqueeze(0).to(self.device)], dim=0)
                rtgs += [rtgs[-1] + reward]##rtgs是负的
                #action = sampled_action.cpu().numpy()[0,-1]
                input_actions += [sampled_action]
            reward_DT_list.append(reward_DT)
        self.model.train(True)
        return np.mean(reward_DT_list)

    def test_returns(self, args, stream, logger, device, ret):
        model_save = torch.load(stream+'BestNet.pt')
        
        model_save.train(False)
        
        env = Env(args,device)
        env.eval()

        T_rewards, T_Qs = [], []
        for i in range(50):
            state = env.reset()
            state = state.unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(model_save.module, state, 1, temperature=1.0, sample=True, actions=None, 
                                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))
            
            j = 0
            all_states = state
            actions = []
            while True:
                if j == 0:
                    reward_sum = 0
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward = env.step(state,action)
                reward_sum += reward
                j += 1

                if j == args.max_episode_length-1:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)
                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                
                sampled_action = sample(model_save.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
            logger.info("test reward DT: %f", reward_sum)


class Env():
    def __init__(self, args, device):
        self.device = device

        self.h = args.h
        self.p = args.p
        self.Demand_Max = args.Demand_Max
        self.mu = args.mu
        self.LT_min = args.LT_min
        self.LT_max = args.LT_max
        self.initial_state = np.ones(args.LT_max+1)*4
        
        actions = np.arange(20,dtype=int)
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        
        self.training = True  # Consistent with model training mode
        
        self.q_arrivals = np.arange(len(self.initial_state)-1)
        
        demand_realizations = np.arange(self.Demand_Max + 1)
        self.demand_probabilities = poisson.pmf(np.arange(self.Demand_Max + 1), mu=self.mu)
        self.demand_probabilities[-1] += 1 - np.sum(self.demand_probabilities)

    def reset(self):
        # Process and return "initial" state
        self.q_arrivals = np.arange(len(self.initial_state)-1)
        return torch.tensor(self.initial_state, dtype=torch.float32, device=self.device).div_(100)

    def step(self, state, action):
        demand = np.random.choice(np.arange(len(self.demand_probabilities)), p=self.demand_probabilities)
        LT = np.random.randint(self.LT_min, self.LT_max+1)
        reward, s1, self.q_arrivals = transition_stochLT(state.cpu().numpy()[0][0]*100, action, demand, self.q_arrivals, LT, self.h, self.p)
        # Return state, reward
        return torch.tensor(s1, dtype=torch.float32, device=self.device).div_(100), reward

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)
        
