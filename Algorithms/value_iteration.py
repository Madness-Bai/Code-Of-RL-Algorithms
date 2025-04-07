import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.arguments import args
from src.grid_world import GridWorld
def value_iteration(env, discount_factor=0.9, theta=1e-6):
    """
    Args:
        env: The environment object.
        discount_foctor: Discount factor for future rewards.
        theta: A small threshold to check for convergence.
    Returns:
        policy: The optimal policy.
        V: The optimal value function.
    """
    # 初始化值函数
    V = np.zeros(env.num_states)
    policy = np.zeros((env.num_states, len(env.action_space)))
    iter = 0
    while True:
        iter += 1
        delta = 0
        # 遍历每个状态 
        for s in range (env.num_states):
            q = []
            v=V[s]
            # Convert state index to coordinates (x, y)
            x = s % env.env_size[0]
            y = s // env.env_size[0]
            state = (x, y)
            for a,action in enumerate (env.action_space):
                next_space, reward = env.get_next_state_and_reward(state, action)
                t = reward + discount_factor * V[next_space]
                q.append(t)
            # 选择最大值
            max_ak = np.argmax(q)
            policy[s,max_ak] = 1
            policy[s,np.arange(len(env.action_space)) != max_ak] = 0
            # 更新值函数
            V[s]=np.max(q)
            delta = max(delta, abs(v - V[s]))
        print(f"Iteration {iter}, delta: {delta}")
        if delta < theta:
            break
    return policy, V

if __name__ == "__main__":
        env = GridWorld()
        env.reset()               
        policy_matrix,values = value_iteration(env)
        if isinstance(env, GridWorld):
            env.render()
            env.add_policy(policy_matrix)
            env.add_state_values(values)
            # Render the environment
            env.render(animation_interval=2)