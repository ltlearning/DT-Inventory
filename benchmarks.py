import numpy as np
import random
from scipy.stats import poisson
from LS_env import transition_stochLT
import argparse
import time,os

def test_bs(args, stream, bs_list):
    LT_min = args.LT_min
    LT_max = args.LT_max
    Demand_Max = args.Demand_Max
    h = args.h
    p = args.p

    best_reward_bs_list = []
    best_cost = 999999
    for bs in bs_list:
        demand_realizations = np.arange(Demand_Max + 1)
        demand_probabilities = poisson.pmf(np.arange(Demand_Max + 1), mu=args.mu)
        demand_probabilities[-1] += 1 - np.sum(demand_probabilities)
        
        reward_bs_list = []
        for k in range(50):
            state = np.ones(LT_max+1)*4
            q_arrivals = np.arange(LT_max)
            reward_bs = 0
            for i in range(90):
                demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
                LT = np.random.randint(LT_min, LT_max+1)
                
                current_action = max(bs-state[0],0)
                
                r, s1,q_arrivals = transition_stochLT(state, current_action, demand, q_arrivals, LT, h, p)
                
                reward_bs += r
                state = s1
            
            reward_bs_list.append(reward_bs)
        print('test bs mean: ',np.mean(reward_bs_list))
        if best_cost>np.mean(reward_bs_list):
            best_cost = np.mean(reward_bs_list)
            best_reward_bs_list = reward_bs_list
    np.save(stream+'test_bs.npy',np.array(best_reward_bs_list))
    print('test best bs mean: ',np.mean(np.array(best_reward_bs_list)))

def test_bs_cap(args, stream, bs_list, cap_list):
    LT_min = args.LT_min
    LT_max = args.LT_max
    Demand_Max = args.Demand_Max
    h = args.h
    p = args.p
    
    best_reward_bs_list = []
    best_cost = 999999
    for bs in bs_list:
        for cap in cap_list:
            demand_realizations = np.arange(Demand_Max + 1)
            demand_probabilities = poisson.pmf(np.arange(Demand_Max + 1), mu=args.mu)
            demand_probabilities[-1] += 1 - np.sum(demand_probabilities)
            
            reward_bs_list = []
            for k in range(50):
                state = np.ones(LT_max+1)*4
                q_arrivals = np.arange(LT_max)
                reward_bs = 0
                for i in range(90):
                    demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
                    LT = np.random.randint(LT_min, LT_max+1)
                    
                    current_action = np.clip(max(bs-state[0],0),0,cap)
                    
                    r, s1,q_arrivals = transition_stochLT(state, current_action, demand, q_arrivals, LT, h, p)
                    
                    reward_bs += r
                    state = s1
                
                reward_bs_list.append(reward_bs)
            print('test bs cap mean: ',np.mean(reward_bs_list))
            if best_cost>np.mean(reward_bs_list):
                best_cost = np.mean(reward_bs_list)
                best_reward_bs_list = reward_bs_list
    np.save(stream+'test_bs_cap.npy',np.array(best_reward_bs_list))
    print('test best bs cap mean: ',np.mean(np.array(best_reward_bs_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--LT_min', type=int, default=3)
    parser.add_argument('--LT_max', type=int, default=5)
    parser.add_argument('--Demand_Max', type=int, default=20)
    parser.add_argument('--mu', type=int, default=8)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--p', type=int, default=5)
    args = parser.parse_args()
    name = 'DT' + time.strftime('%Y-%m-%d-%H-%M-%S')
    stream = os.path.join('DTlog', name)
    bs_list=np.arange(5,18)
    cap_list=np.arange(10,18)
    test_bs(args, stream, bs_list)
    test_bs_cap(args, stream, bs_list, cap_list)