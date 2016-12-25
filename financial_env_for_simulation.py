#coding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import tensorflow as tf


class simulationEnv(gym.Env):

    def __init__(self,opt):
        self._opt=str_to_dic(opt)
        args=self._opt
        self.points = int(args['points']) or 10 #数据点
        self.dt = args['dt'] or 0.05  #间隔
        self.sin_index = args['sin_index'] or 0
        self.hold_num = args['hold_num'] or 0
        self.noise = args['noise'] or 0
        self.Account_All = args['Account_All'] or 100
        self.lossRate = args['lossRate'] or 0.6
        self.Account = self.Account_All
        self.max = args['max'] or 100
        self.lever = 1
        self.cost = 0
        self.price = []
        self.sindex = []
        self.shb = []
        self.trw = 0
        self.own = []
        self.action_index = []
#为price预存points个点
        for i in xrange(0,self.points):
            self.price.append(self._get_TestData(i - self.points+1, self.dt))
            self.sindex.append(i - self.points+1)

        #self._seed()
        self._action_set = [-1,0,1]
        self.action_space = spaces.Discrete(len(self._action_set))

    def _get_TestData(self,sin_index,dt):
        x=np.random.uniform(0,self.noise)
        y=np.power(-1,np.random.random_integers(1,100))
        return np.abs(np.sin(sin_index*dt+0.001)+1+self.noise+x*y)

    def step(self, a):
        reward = 0.0
        terminal=False
        action = self._action_set[a]

        self.sin_index = self.sin_index + 1
        sin_index = self.sin_index
        points = self.points
        dt = self.dt
        self.shb.append(action + 1 + 1)
        self.sindex.append(sin_index)
        self.action_index.append(sin_index)
#next price point
        self.price.append(self._get_TestData(sin_index, dt))
#差价
        dprice = self.price[sin_index + points-1] - self.price[sin_index + points - 1-1]

        if action == -1:
            if self.hold_num <= 0:
                self.hold_num = self.hold_num + action
            if self.hold_num > 0:
                action = action * np.abs(self.hold_num)
                self.hold_num = 0
        if action == 1:
            if self.hold_num < 0:
                action = action * np.abs(self.hold_num)
                self.hold_num = 0
            if self.hold_num >= 0:
                self.hold_num = self.hold_num + action
#reward except cost
        rw = self.hold_num * dprice * self.lever - self.cost * np.abs(action)
        self.trw = self.trw + rw
        self.own.append(self.trw / self.max)
        self.Account = self.Account - action * self.price[sin_index + points-1] - self.cost * np.abs(action)

        sinTensor=self.price[sin_index:sin_index+points]
        sinTensor.append(self.hold_num)
        tmp = self.Account + self.hold_num * self.price[sin_index + points-1]
        sinTensor.append(tmp)
        if tmp < self.Account_All * (1 - self.lossRate):
            terminal = True
        sinTensor = np.asarray(sinTensor)
        return sinTensor.reshape(points+2,1), rw, terminal, {}

    # return: (states, observations)
    def reset(self):
        self.hold_num=0
        self.Account=self.Account_All
        return self.step(0)

    def get_obs(self):
        obs=np.asarray(self.price[0:self.points+2])
        return obs.reshape(self.points+2,1)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    # def save_state(self):
    #     return self.ale.saveState()

    # def load_state(self):
    #     return self.ale.loadState()

    # def clone_state(self):
    #     return self.ale.cloneState()

    # def restore_state(self, state):
    #     return self.ale.restoreState(state)

def str_to_dic(args):
    opt={}
    tmp=args.split(',')
    for param in tmp:
        key=param.split('=')[0]
        value=param.split('=')[1]
        opt[key]=float(value)
    return opt

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
