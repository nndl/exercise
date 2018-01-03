import gym
import random
import numpy as np

from RL_QG_agent import RL_QG_agent

env = gym.make('Reversi8x8-v0')
env.reset()

agent = RL_QG_agent()
agent.load_model()

max_epochs = 100

for i_episode in range(max_epochs):
    observation = env.reset()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    for t in range(100):
        action = [1,2]
        # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

        ################### 黑棋 ############################### 0表示黑棋
        #  这部分 黑棋 是随机下棋
        env.render()  #  打印当前棋局
        enables = env.possible_actions
        if len(enables) == 0:
            action_ = env.board_size**2 + 1
        else:
            action_ = random.choice(enables)
        action[0] = action_
        action[1] = 0   # 黑棋 为 0
        observation, reward, done, info = env.step(action)
        ################### 白棋 ############################### 1表示白棋
        env.render()
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1 # pass
        else:
           action_  = agent.place(observation, enables) # 调用自己训练的模型
        action[0] = action_
        action[1] = 1  # 白棋 为 1
        observation, reward, done, info = env.step(action)


        if done: # 游戏 结束
            print("Episode finished after {} timesteps".format(t+1))
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            if black_score >32:
                print("黑棋赢了！")
            else:
                print("白棋赢了！")
            print(black_score)
            break