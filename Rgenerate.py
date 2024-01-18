import numpy as np
import random
from scipy.stats import poisson
from LS_env import transition_stochLT


def generate_data(args):
    LT_min = args.LT_min
    LT_max = args.LT_max
    Demand_Max = args.Demand_Max
    demand_realizations = np.arange(Demand_Max + 1)
    demand_probabilities = poisson.pmf(np.arange(Demand_Max + 1), mu=args.mu)
    demand_probabilities[-1] += 1 - np.sum(demand_probabilities)
    h = args.h
    p = args.p
    
    state_bs_list = []
    action_bs_list = []
    reward_bs_list = []
    demand_bs_list = []
    for k in range(100):
        state = np.ones(LT_max+1)*4
        q_arrivals = np.arange(LT_max)
        state_bs = []
        action_bs = []
        reward_bs = []
        demand_bs = []
        for i in range(90):
            demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
            demand_bs.append(demand)
            LT = np.random.randint(LT_min, LT_max+1)
            
            current_action = np.random.randint(0,17)
            
            r, s1,q_arrivals = transition_stochLT(state, current_action, demand, q_arrivals, LT, h, p)
            
            action_bs.append(current_action)
            state_bs.append(state)
            reward_bs.append(r)
            state = s1
        
        state_bs_list.append(state_bs)
        action_bs_list.append(action_bs)
        reward_bs_list.append(reward_bs)
        demand_bs_list.append(demand_bs)
    np.save('lost_sales/action.npy',np.array(action_bs_list))
    np.save('lost_sales/state.npy',np.array(state_bs_list))
    np.save('lost_sales/cost.npy',np.array(reward_bs_list))#best: 3185
    np.save('lost_sales/demand.npy',np.array(demand_bs_list))
    print('generate data:',np.mean(np.array(reward_bs_list).sum(1)))
        