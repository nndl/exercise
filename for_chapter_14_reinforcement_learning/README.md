# 黑白棋游戏



## 问题描述：

​	本次作业要实现的是 利用强化学习等知识 玩atari 游戏中的 黑白棋游戏。具体任务是补全RL_QG_agent.py 文件.





## 环境配置: 

配置环境安装办法：

1. pip install gym[all] # 安装 gym
2. 找到 安装的包的目录，然后复制 github 上面的reversi 文件夹， 到gym/envs/ 中
  （windows中的目录路径是
  C:\Program Files\Anaconda3\Lib\site-packages\gym\envs）
3. 在envs文件夹中 有__init__.py 文件，在文件 末尾，添加注册信息。
  （参考 github 上面 __init__.py 文件末尾的注册信息（即id='Reversi8x8-v0', 的注册信息））





## 题目要求： 

​	Github 中reversi_main.py 是一个demo程序，主要为了规范后期判作业时候的接口.本作业后面会运行大家的程序，因此需要统一接口，并且注意保证自己的代码没有错误，可以运行。训练程序的时候 黑白双方可以自己规定，环境中没有对弈对象。因此训练程序的时候时自己设置对弈对象，比如与随机进行对弈，其次可以和一些搜索算法对弈。



参考文献：

1. Learning to Play Othello with Deep Neural Networks
2. Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play

