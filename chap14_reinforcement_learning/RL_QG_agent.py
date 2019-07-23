import tensorflow as tf
import os

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi")
        pass    # 删掉这句话，并填写相应代码

    def init_model(self):

        # 定义自己的 网络
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        # 补全代码


    def place(self,state,enables):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        action = 123456789    # 删掉这句话，并填写相应代码

        return action

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    # 定义自己需要的函数