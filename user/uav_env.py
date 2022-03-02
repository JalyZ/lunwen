import logging
import numpy as np
import gym

logger = logging.getLogger(__name__)

N = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

M = np.zeros([20,20],dtype=int)-1
M[4,5] = 0
M[14,10] = 0
M[7,15] = 0

class PathEnv1(gym.Env):
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
        self.state = self.state + np.dot(action,transMatrix)
        for s in self.state:
            if min(s[0],s[1])<0 or max(s[0],s[1])>19:
                s[0] = np.clip(s[0],0,19)
                s[1] = np.clip(s[1],0,19)
                self.r += -10
    

    def reset(self):
        self.state = np.array([[0,0],[19,0],[0,19]])
        self.r = 0
        self.step_num = 0
        return self.state

    
    def reward(self):
        r = - 1
        for s in self.state:
            r = M[s[0],s[1]]
        self.r = r

    def judge(self):
        if self.step_num < 3000:
            if np.array([4,5]) in self.state:
                if np.array([14,10]) in self.state:
                   if np.array([7,15]) in self.state:
                       return True
            return False
        return True
    
    def step(self, action):
        self.__apply_action(action)
        self.step_num += 1
        state = self.state
        self.reward()
        if self.judge():
            done = True
        else:
            done = False
        info = {}
        return state, self.r, done, info
    
    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass