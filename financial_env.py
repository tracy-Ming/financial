#coding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import tensorflow as tf


class financialEnv(gym.Env):

    def __init__(self,opt,filepath):
        self._opt=str_to_dic(opt)
        args=self._opt
        self.points = int(args['points']) or 10 #数据点
        self.fx_index = 0
        self.hold_num = args['hold_num'] or 0
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
        self.rw=[]
        self.action_index = []
        self.maxdown=100
        self.maxdown_action=0
        self.data,self.data_num=self.dataloading(filepath)
#为price预存points个点
        for i in xrange(0,self.points):
            self.price.append(self.data[i])
            self.sindex.append(i - self.points+1)
        #print self.price[0]
        #self._seed()
        self._action_set = [-1,0,1]
        self.action_space = spaces.Discrete(len(self._action_set))

    def dataloading(self,filepath):
        csvfile = np.loadtxt(filepath, dtype=np.str, delimiter=',')
        data = csvfile[0:, 2:].astype(np.float32)
        num=np.asarray(data).__len__()
        return np.asarray(data)[:,0],num

    def step(self, a):
        reward = 0.0
        terminal=False
        action = self._action_set[a]

        self.fx_index = self.fx_index + 1
        fx_index = self.fx_index
        points = self.points
        self.shb.append(action + 1 + 1)
        self.sindex.append(fx_index)
        self.action_index.append(fx_index)
#next price point
        data_index=fx_index+points-1
        if data_index>self.data_num-1:
            data_index=data_index%self.data_num
        self.price.append(self.data[data_index])
#差价
        dprice = self.price[fx_index + points-1] - self.price[fx_index + points - 1-1]

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
        self.Account = self.Account - action * self.price[fx_index + points-1] - self.cost * np.abs(action)

        sinTensor=self.price[fx_index:fx_index+points]
        sinTensor.append(self.hold_num)
        tmp = self.Account + self.hold_num * self.price[fx_index + points-1]
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
