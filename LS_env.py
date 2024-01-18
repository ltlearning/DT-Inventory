import numpy as np
import itertools


def actions(args):
    actions = np.arange(args.max_order)
    return actions
#先demand,cost再收货
def transition_stochLT(state, a, d, q_arrivals, LT, h, p):
    #print(state, a, d, q_arrivals, LT)
    #breakpoint()
    s=state.copy()
    
    #s1[0] = np.clip(s[0] - d, 1-args.inv_max, args.inv_max - 1)
    reward = max(s[0] - d, 0) * h + min(s[0] - d, 0) * -p#lost sales
    
    s[0] = max(s[0] - d,0)
    arrived = np.sum(s[1:][q_arrivals==0])
    s[1:][q_arrivals==0] = 0
    s[0] += arrived
    s1 = np.roll(s, -1)
    s1[0] = s[0]
    s1[-1] = a

    q_arrivals -= 1
    q_arrivals = np.roll(q_arrivals,-1)
    q_arrivals[-1] = LT-1
    q_arrivals = np.clip(q_arrivals,0,np.inf)
    
    return reward, s1, q_arrivals