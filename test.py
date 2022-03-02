import gym
import numpy as np
import time

from gym import spaces

M = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])


class CircleDrive(gym.Env):
    def __init__(self, render : bool = False):
        self._render = render
        # 定义动作空间
        self.action = np.zeros([3,5],dtype=int)

        # 定义状态空间
        self.state = np.zeros([3,2],dtype=int)

        # 定义回报
        self.r = 0

        # 计数器
        self.step_num = 0
    
    def __apply_action(self, action):
        transMatrix = np.array([[0,1],[0,-1],[-1,0],[1,0],[0,0]])
        self.state = self.state + action*transMatrix
        for s in self.state:
            if min(s[0],s[1])<0:
                self.r += -10
    

    def reset(self):
        self.state = np.array([[0,0],[20,0],[0,20]])
        return self.state

    
    def reward(self):
        r = - 1
        for s in self.state:
            r += M[s[0],s[1]]
        self.r += r

    def step(self, action):
        self.__apply_action(action)
        self.step_num += 1
        state = self.state
        self.reward()
        if self.step_num > 36000:
            done = True
        else:
            done = False
        info = {}
        return state, self.r, done, info
    
    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass
    


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env 
    # 如果你安装了pytorch，则使用上面的，如果你安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
    env = CircleDrive()
    check_env(env)


""" env = gym.make('PathEnv-v1')
env.reset()
env.render()
time.sleep(1)
env.close() """