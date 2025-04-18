import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from examples.arguments import args
from src.grid_world import GridWorld

def MC_Exploring_Starts(env,num_episodes=10000,discount_factor=0.9,iteration=1000):
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    Q = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
    for i in range(iteration):
        return_temp = np.zeros((env.num_states, len(env.action_space)))
        state_action_pair = []
        reward = []
        # Initialize return_temp as a numpy array with the same shape as Q
        start_state= i % (env.num_states*len(env.action_space))
        state = start_state // env.num_states
        action = start_state % len(env.action_space)
        state_action_pair.append((state, action))
        temp_state,temp_reward = env.get_next_state_and_reward((state % env.env_size[0], state // env.env_size[0]), env.action_space[action])
        reward.append(temp_reward)
        for j in range(num_episodes):
            action_idx = np.argmax(policy[temp_state])
            action = env.action_space[action_idx]
            state_action_pair.append((temp_state, action_idx))
            next_state,next_reward = env.get_next_state_and_reward((temp_state % env.env_size[0], temp_state // env.env_size[0]), action)
            temp_state = next_state
            reward.append(next_reward)
        g = 0
        # 截止目前已经得到了一个episode的state action pair和reward，接下来要倒叙遍历并赋值
        for t in range(len(state_action_pair)-1,-1,-1):
            g = discount_factor * g + reward[t]
            # use the first_visit method:
            if (state_action_pair[t] not in state_action_pair[:t]):
                s , a = state_action_pair[t]
                return_temp[s,a] = g
                Q[s,a] = return_temp[s,a]
            else:
                continue
        # policy improvement
        for s in range(env.num_states):
            action_idx = np.argmax(Q[s])
            policy[s,action_idx] = 1
            policy[s,np.arange(len(env.action_space))!=action_idx] = 0
            V[s] = max(Q[s])
        print(f"Iteration {i}, policy: {policy}, V: {V}")
    return policy, V

if __name__ == "__main__":
        env = GridWorld()
        env.reset()               
        policy_matrix,values = MC_Exploring_Starts(env)
        if isinstance(env, GridWorld):
            env.render()
            print("policy_matrix",policy_matrix)
            
            # Define the expected converged policy matrix
            expected_policy = np.array([
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [1., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1.]
            ])
            
            # Check if policy has converged
            if np.array_equal(policy_matrix, expected_policy):
                print("Policy has converged to the expected optimal policy!")
            
            env.add_policy(policy_matrix)
            env.add_state_values(values)
            # Render the environment
            env.render(animation_interval=2)
            

    