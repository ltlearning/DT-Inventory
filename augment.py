import numpy as np

def bs_cap_augment(bs_list,cap_list):
    actions = np.load('lost_sales/action.npy')[:-10]
    stepwise_returns = np.load('lost_sales/cost.npy')[:-10]
    demand_list = np.load('lost_sales/demand.npy')[:-10]
    obss = np.load('lost_sales/state.npy')[:-10]
    
    returns = np.array(stepwise_returns.sum(1))
    historical_data_mean = np.mean(returns)
    
    best_bs = 0
    best_cap = 0
    best_cost = 99999999999
    for bs in bs_list:
        for cap in cap_list:
            state_bs_list = []
            action_bs_list = []
            reward_bs_list = []
            for k in range(len(returns)):#50个horizon
                state = obss[k]
                demand = demand_list[k]
                state_bs = []
                action_bs = []
                reward_bs = []
                previous_state = state[0]
                for i in range(90):
                    state_bs.append(previous_state)
                    current_demad = demand[i]

                    reward_bs.append(max(previous_state[0] - current_demad, 0) * 1 + min(previous_state[0] - current_demad, 0) * -5)

                    current_state = []
                    current_inv = previous_state[0]
                    current_action = np.clip(max(bs-current_inv,0),0,cap)#####bs_cap policy
                    action_bs.append(current_action)
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
                state_bs_list.append(state_bs)
                action_bs_list.append(action_bs)
                reward_bs_list.append(reward_bs)
                if np.mean(reward_bs)<best_cost:
                    best_cost = np.mean(np.array(reward_bs_list).sum(1))
                    best_bs = bs
                    best_cap = cap
            #print('bs cap augment cost:',np.mean(np.array(reward_bs_list).sum(1)))
            if np.mean(np.array(reward_bs_list).sum(1))<historical_data_mean:
                np.save('lost_sales/cap/action_bs'+str(bs)+'cap'+str(cap)+'.npy',np.array(action_bs_list))
                np.save('lost_sales/cap/state_bs'+str(bs)+'cap'+str(cap)+'.npy',np.array(state_bs_list))
                np.save('lost_sales/cap/cost_bs'+str(bs)+'cap'+str(cap)+'.npy',np.array(reward_bs_list))
    return best_bs,best_cap

def bs_augment(bs_list):
    actions = np.load('lost_sales/action.npy')[:-10]
    stepwise_returns = np.load('lost_sales/cost.npy')[:-10]
    demand_list = np.load('lost_sales/demand.npy')[:-10]
    obss = np.load('lost_sales/state.npy')[:-10]
    
    returns = np.array(stepwise_returns.sum(1))
    historical_data_mean = np.mean(returns)
    
    best_bs = 0
    best_cost = 99999999999
    for bs in bs_list:
        state_bs_list = []
        action_bs_list = []
        reward_bs_list = []
        for k in range(len(returns)):#50个horizon
            state = obss[k]
            demand = demand_list[k]
            state_bs = []
            action_bs = []
            reward_bs = []
            previous_state = state[0]
            for i in range(90):
                state_bs.append(previous_state)
                current_demad = demand[i]

                reward_bs.append(max(previous_state[0] - current_demad, 0) * 1 + min(previous_state[0] - current_demad, 0) * -5)

                current_state = []
                current_inv = previous_state[0]
                current_action = max(bs-current_inv,0)#####bs policy
                action_bs.append(current_action)
                if i == 89:
                    break
                
                #current_state的变化跟previous_state和current_demad，current_action以及state有关
                current_state += [max(current_inv - current_demad,0) + previous_state[1]]#第一位
                for j in range(1,len(state[i+1])-1):#根据下一个state推测订单的到达时间
                    if state[i+1][j]==0:
                        current_state[0] += previous_state[j+1]#如果下一个state某个位置j为0，说明当前state的j+1位置订单会在这个时候到达
                        current_state.append(0)#那么j位置的订单变成0
                    else:
                        current_state.append(previous_state[j+1])#没有到达的话j+1位置的订单变成j位置的订单
                current_state.append(current_action)#最后一位是当前的订单
                previous_state = current_state
            state_bs_list.append(state_bs)
            action_bs_list.append(action_bs)
            reward_bs_list.append(reward_bs)
            if np.mean(reward_bs)<best_cost:
                best_cost = np.mean(np.array(reward_bs_list).sum(1))
                best_bs = bs
        #print('bs augment cost:',np.mean(np.array(reward_bs_list).sum(1)))
        if True:#
        #if np.mean(np.array(reward_bs_list).sum(1))<historical_data_mean:
            np.save('lost_sales/action_bs'+str(bs)+'.npy',np.array(action_bs_list))
            np.save('lost_sales/state_bs'+str(bs)+'.npy',np.array(state_bs_list))
            np.save('lost_sales/cost_bs'+str(bs)+'.npy',np.array(reward_bs_list))
    return best_bs