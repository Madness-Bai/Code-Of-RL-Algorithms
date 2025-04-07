import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.arguments import args
from src.grid_world import GridWorld

def policy_iteration(env,discount_factor=0.9,theta=1e-6,epochs=10):
    V = np.zeros(env.num_states)
    policy = np.random.uniform(0,1,(env.num_states, len(env.action_space)))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    iter = 0
    while True:
        iter += 1
        delta = 0
        temp_V = V.copy()
        # Policy Evaluation
        for i in range(epochs):
            # 计算当前状态的总value
            for s in range(env.num_states):
                q = []
                state = (s % env.env_size[0], s // env.env_size[0])
                for j, action in enumerate(env.action_space):
                    next_space, reward = env.get_next_state_and_reward(state, action)
                    t = reward + discount_factor * V[next_space]
                    q.append(t)
                if s==0:
                    print(f"current epoch: {epochs} q: {q},policy: {policy[s]}")
                V[s]= np.dot(policy[s], np.array(q))
        # Policy Improvement
        for s in range(env.num_states):
            q_value=[]
            state = (s % env.env_size[0], s // env.env_size[0])
            for j, action in enumerate(env.action_space):
                next_space, reward = env.get_next_state_and_reward(state, action)
                t = reward + discount_factor * V[next_space]
                q_value.append(t)
            max_ak = np.argmax(q_value)
            policy[s,np.arange(len(env.action_space))!=max_ak] = 0
            policy[s, max_ak] = 1
        delta = max(delta, abs(temp_V[s] - V[s]))
        print(f"Iteration {iter}, delta: {delta}")
        if delta < theta:
            break
    return policy, V
if __name__ == "__main__":
    env = GridWorld()
    env.reset()               
    policy_matrix, values = policy_iteration(env)
    if isinstance(env, GridWorld):
        env.render()
        env.add_policy(policy_matrix)
        env.add_state_values(values)
        # Render the environment
        env.render(animation_interval=2)    




            



