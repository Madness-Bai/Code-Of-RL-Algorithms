import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.arguments import args
from src.grid_world import GridWorld
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 输入的2个维度，代表状态，输出5个维度，5个元素分别是对5个动作（action）估计出来的Q值。
# 在dqn算法中，Q值不是tabuler存储的，而是用神经网络来近似，在需要使用q值时，就把状态输入到神经网络中，得到Q值。
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 5)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def dqn(env, gt_state_values, discount_factor=0.9, c_steps = 10, num_experience = 1000, batch_size = 200, epochs = 1000):
    # 随机策略
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    replay_buffer = []
    # 0-(n-1) 之间选一个状态
    s = np.random.randint(env.num_states)
    visit_history = []
    losses = []
    deltas = []
    # Store the experience samples generated by πb in a replay buffer 
    for e in range(num_experience):
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        next_state, reward = env.get_next_state_and_reward((s % env.env_size[0], s // env.env_size[0]), env.action_space[a])
        replay_buffer.append(((s%env.env_size[0], s//env.env_size[0]), a, reward, (next_state % env.env_size[0], next_state // env.env_size[0])))
        s = next_state
        visit_history.append((s,a))
    main_net, target_net = Net(), Net()
    # 同步两个网络的参数
    target_net.load_state_dict(main_net.state_dict())
    # 创建了一个Adam优化器
    # 将主网络（main_net）中的所有参数传递给优化器，让优化器负责更新这些参数。
    # 此处的参数是神经网络的权重和偏置，在该优化器中会优化所有参数。
    optimizer = optim.Adam(main_net.parameters(), lr=0.01)
    # 创建了一个均方误差损失函数
    # 用于计算预测值和目标值之间的误差（损失），指导优化器如何调整参数。
    criterion = nn.MSELoss()
    for k in range(epochs):
        batch = np.random.choice(len(replay_buffer), size=batch_size)
        states = torch.tensor([replay_buffer[i][0] for i in batch], dtype=torch.float).view(-1, 2)
        actions = torch.tensor([replay_buffer[i][1] for i in batch], dtype=torch.long).view(-1, 1)
        rewards = torch.tensor([replay_buffer[i][2] for i in batch], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor([replay_buffer[i][3] for i in batch], dtype=torch.float).view(-1, 2)
        
        main_net_input = states/env.env_size[0]
        # gather 找出当前s对应的a的Q值
        q_values = main_net(main_net_input).gather(1,actions.long())
        target_net_input = next_states/env.env_size[0]
        with torch.no_grad():
            # 计算目标Q值, max(1)表示在第1维上找到最大值,[0]表示返回最大值,view(-1,1)表示将结果转换为列向量
            target_q_values = target_net(target_net_input).max(1)[0].view(-1,1)
        target = rewards + discount_factor * target_q_values
        # 计算损失
        loss = criterion(q_values, target)
        # 记录损失
        losses.append(loss.item())
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        if k % c_steps == 0:
            target_net.load_state_dict(main_net.state_dict())
        # 计算delta,当前Q值和最优Q值的差值
        delta = 0
        for s in range(env.num_states):
            with torch.no_grad():
                q_values = main_net(torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                                             dtype=torch.float).view(-1,2))
                delta = max(delta, abs(q_values.max().item() - gt_state_values[s]))
        deltas.append(delta)

    policy = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
    for s in range(env.num_states):
        with torch.no_grad():
            q_values = main_net(torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                                             dtype=torch.float).view(-1,2))
        V[s] = torch.max(q_values).item()
        max_idx = torch.argmax(q_values).item()
        policy[s, max_idx] = 1

    plt.subplots(3, 1, figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.hist([i[0] for i in visit_history])
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.subplot(3, 1, 3)
    plt.plot(deltas)
    plt.title('Value Error')
    plt.show()
    return policy, V

if __name__ == "__main__":
    gt_state_values = [18.697814, 21.88646, 25.4294, 29.366, 33.74, 21.88646, 25.4294, 29.366, 33.74, 38.6, 25.4294, 29.366, 33.74, 38.6, 44.0, 29.366, 33.74, 38.6, 44.0, 50.0, 33.74, 38.6, 44.0, 50.0, 50.0]
    env = GridWorld()
    env.reset()               
    policy_matrix,values = dqn(env, gt_state_values=gt_state_values)
    if isinstance(env, GridWorld):
        env.render()
        print("policy_matrix",policy_matrix)
        env.add_policy(policy_matrix)
        env.add_state_values(values)
        # Render the environment
        env.render(animation_interval=2)
        
                


        
        