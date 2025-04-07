__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import sys   
import os 
sys.path.append("..")         
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches          
from examples.arguments import args           

class GridWorld():

    def __init__(self, env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):

        # 环境大小
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        # 进入禁止状态的奖励
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        self.canvas = None
        # 动画间隔时间
        self.animation_interval = args.animation_interval


        self.color_forbid = (0.9290,0.6940,0.125)
        self.color_target = (0.3010,0.7450,0.9330)
        self.color_policy = (0.4660,0.6740,0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0,0,1)



    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state] 
        return self.agent_state, {}


    def step(self, action):
        # 验证 action 是否在 self.action_space 中
        assert action in self.action_space, "Invalid action"

        next_state, reward  = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        # 在下一个状态基础上增加了一些随机噪声和动作的影响
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action))
        # 下一个状态本身
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}   
    
        
    def _get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0,-1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        # 保持原状态不进入禁止状态
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
        

    def _is_done(self, state):
        return state == self.target_state
    
    # 创建了一个可视化的网格世界，显示了智能体的位置、轨迹、目标状态和禁止状态，并支持动态更新。
    def render(self, animation_interval=args.animation_interval, save_final=True):
        if self.canvas is None:
            # 开启了交互模式，允许实时更新图形而不阻塞程序执行。
            plt.ion()
            # 创建新的图形窗口（Figure）和坐标轴（Axes）对象。                             
            self.canvas, self.ax = plt.subplots()  
            # 设置了 x 轴和 y 轴的显示范围。
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            # 设置了 x 轴和 y 轴的刻度范围。
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))
            # 添加网格线     
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')     
            # 设置了坐标轴的比例为相等，这样 x 轴和 y 轴的单位长度是一样的。     
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()                           
            self.ax.xaxis.set_ticks_position('top')           
            
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')           
            for lb in idx_labels_y:
                self.ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,labeltop=False)   
            # 添加目标状态和禁止状态
            self.target_rect = patches.Rectangle( (self.target_state[0]-0.5, self.target_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)     

            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle((forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_forbid, facecolor=self.color_forbid)
                self.ax.add_patch(rect)
            # 创建智能体和轨迹对象
            self.agent_star, = self.ax.plot([], [], marker = '*', color=self.color_agent, markersize=20, linewidth=0.5) 
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)
        #更新智能体位置和轨迹：
        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])       
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)

        # 关闭交互模式，以便程序继续执行。
        if args.debug:
            input('press Enter to continue...')     


    # 在网格世界中直观地展示策略
    def add_policy(self, policy_matrix):
        # 索引和对应值                  
        for state, state_action_group in enumerate(policy_matrix):    
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    # 如果当前动作不是停留，就画一个箭头，否则画一个yuan quan
                    if (dx, dy) != (0,0):
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))
                    else:
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))
    # 显示价值函数或Q值等信息
    def add_state_values(self, values, precision=1):
        '''
            values: iterable
        '''
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(x, y, str(value), ha='center', va='center', fontsize=10, color='black')

    def get_next_state_and_reward(self, state, action):
        next_state, reward = self._get_next_state_and_reward(state, action)
        return next_state[1] * self.env_size[0] + next_state[0], reward