import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.arguments import args
from src.grid_world import GridWorld

# off-policy
def Q_learning(env, num_episodes=10000, discount_factor=0.9, iteration=1000,learning_rate = 0.1,epsilon=0.1):
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    Q = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
    # 画图
    lengths = []
    total_rewards = []

    for i in range(iteration):
        # 初始化state action
        start_state= i % (env.num_states*len(env.action_space))
        state = start_state // env.num_states
        # 根据当前policy选择action,p是概率
        length =0
        total_reward = 0
        for j in range(num_episodes):
            while state != env.num_states - 1:
                action = np.random.choice(np.arange(len(env.action_space)), p=policy[state])
                next_state, next_reward = env.get_next_state_and_reward((state % env.env_size[0], state // env.env_size[0]), env.action_space[action])
                Q[state,action] = Q[state,action] - learning_rate * (Q[state,action]-(next_reward+discount_factor* max(Q[next_state])))
                # update policy
                action_idx = np.argmax(Q[state])
                policy[state,action_idx] = 1 - ((len(env.action_space)-1) * epsilon)/len(env.action_space)
                policy[state,np.arange(len(env.action_space))!=action_idx] = (epsilon)/len(env.action_space)
                V[state] = np.sum(policy[state] * Q[state])
                state = next_state
                length += 1
                total_reward += next_reward
        lengths.append(length)
        total_rewards.append(total_reward)

    # TODO: plot the graph of the convergence of the length of episodes and that of the total rewards of episodes
    fig = plt.subplots(2, 1)
    plt.subplot(2, 1, 1)
    plt.plot(lengths)
    plt.xlabel('Iterations')
    plt.ylabel('Length of episodes')
    plt.subplot(2, 1, 2)
    plt.plot(total_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Total rewards of episodes')
    plt.show()
        #print(f"Iteration {i}, policy: {policy}, V: {V}")
    return policy, V

if __name__ == "__main__":
    env = GridWorld()
    env.reset()               
    policy_matrix,values = Q_learning(env)
    if isinstance(env, GridWorld):
        env.render()
        print("policy_matrix",policy_matrix)
        env.add_policy(policy_matrix)
        env.add_state_values(values)
        # Render the environment
        env.render(animation_interval=2)