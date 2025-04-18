import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from examples.arguments import args
from src.grid_world import GridWorld

def MC_Basic(env,num_episodes=100,discount_factor=0.9):
    V = np.zeros(env.num_states)
    # policy = np.zeros((env.num_states, len(env.action_space)), dtype=int)
    # policy[:, 0] = 1
    policy = np.eye(5)[np.random.randint(0, 5, size=(env.env_size[0] * env.env_size[1]))]
    for e in range(1000):
        for s in range (env.num_states):
            state = (s% env.env_size[0], s//env.env_size[0])
            q=[]
            for a , action in enumerate(env.action_space):
                reward = 0
                action_iter=action
                # print(action_iter)
                # print(state)
                total_reward = 0
                for i in range(num_episodes):
                    next_space, reward = env._get_next_state_and_reward(state, action_iter)
                    total_reward += np.power(discount_factor, i) * reward
                    state = next_space
                    next_s= next_space[1] * env.env_size[0] + next_space[0]
                    next_action_num=np.argmax(policy[next_s])
                    action_iter = env.action_space[next_action_num]
                    print("action_iter: ",action_iter)
                q.append(total_reward)
            max_idx= np.argmax(q)
            policy[s,max_idx]=1
            policy[s,np.arange(len(env.action_space))!=max_idx]=0
            V[s] = max(q)
    return policy, V

if __name__ == "__main__":
        env = GridWorld()
        env.reset()               
        policy_matrix,values = MC_Basic(env)
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
        
# 很神奇，最后还是没有收敛到所有状态都达到最优策略
