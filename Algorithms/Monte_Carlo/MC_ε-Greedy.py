import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from examples.arguments import args
from src.grid_world import GridWorld

def MC_ε_Greedy(env,num_episodes=100,discount_factor=0.9,iteration=10000,epsilon=0.1):
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    Q = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
   
    for i in range(iteration):
        return_temp = np.zeros((env.num_states, len(env.action_space)))
        count_temp = np.zeros((env.num_states, len(env.action_space)))
        state_action_pair = []
        reward = []
        # Initialize return_temp as a numpy array with the same shape as Q
        state = 0
        a = np.random.choice(len(env.action_space), p=policy[state])
        # state_action_pair.append((state, a))
        # temp_state,temp_reward = env.get_next_state_and_reward((state % env.env_size[0], state // env.env_size[0]), env.action_space[a])
        # reward.append(temp_reward)
        s_temp = state
        for j in range(num_episodes):
            action = env.action_space[a]
            state_action_pair.append((s_temp, a))
            next_state,next_reward = env.get_next_state_and_reward((s_temp % env.env_size[0], s_temp // env.env_size[0]), action)
            s_temp = next_state
            reward.append(next_reward)
            a = np.random.choice(len(env.action_space), p=policy[s_temp])
        g = 0
        # 截止目前已经得到了一个episode的state action pair和reward，接下来要倒叙遍历并赋值
        for t in range(len(state_action_pair)-1,-1,-1):
            g = discount_factor * g + reward[t]
            # use the every_visit method:
            s , a = state_action_pair[t]
            return_temp[s,a] += g
            count_temp[s,a] += 1
            Q[s,a] = return_temp[s,a] / count_temp[s,a] if count_temp[s,a] > 0 else 0
        for s in range(env.num_states):
            action_idx = np.argmax(Q[s])
            policy[s,action_idx] = 1 - ((len(env.action_space)-1) * epsilon)/len(env.action_space)
            policy[s,np.arange(len(env.action_space))!=action_idx] = (epsilon)/len(env.action_space)
            V[s] = max(Q[s])
        print(f"Iteration {i}, policy: {policy}, V: {V}")
    return policy, V

if __name__ == "__main__":
        env = GridWorld()
        env.reset()               
        policy_matrix,values = MC_ε_Greedy(env)
        if isinstance(env, GridWorld):
            env.render()
            print("policy_matrix",policy_matrix)
            env.add_policy(policy_matrix)
            env.add_state_values(values)
            # Render the environment
            env.render(animation_interval=2)
            

    