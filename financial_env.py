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
        self.points = int(args['points']) or 100 #数据点
        self.fx_index = 0
        self.hold_num = args['hold_num'] or 0
        self.Account_All = args['Account_All'] or 3000
        self.lossRate = args['lossRate'] or 0.6
        self.Account = self.Account_All
        self.max = args['max'] or 100
        minimum_order=1000 #### usually minimum order is 0.01 lot size =1000
        self.lever = 500*minimum_order ####because action unit is 1, so I combine minmum_order with leverage
        self.cost = 0.035
        self.swap=0.01/60/24/self.lever*(500*1000)  ### each day interest rate is 0.01 for 0.01 lot and 500 leverage , and here it is the rate for each minute
        self.price = []
        self.trw = 0
        self.rw=[]
        self.lastactiontime=0
        self.action_index = []
        self.maxdown=100
        self.maxdown_action=0
        self.data,self.data_num=self.dataloading(filepath)
#plot
        self.price_data=[]#plot price
        self.action_data=[]#plot action
        self.treward_data=[]#plot total reward

#为price预存points个点
        for i in xrange(0,self.points):
            self.price.append(self.data[i])
            tmp = []
            tmp.append(i - self.points+1)
            tmp.append(self.data[i])
            self.price_data.append(tmp)

        #print self.price[0]
        #self._seed()
        self._action_set = [-1,0,1]  ###define all possible action
        self.action_space = spaces.Discrete(len(self._action_set))


    def dataloading(self,filepath):
        csvfile = np.loadtxt(filepath, dtype=np.str, delimiter=',')
        data = csvfile[0:, 2:].astype(np.float32)
        num=np.asarray(data).__len__()
        return np.asarray(data)[:,0],num

    def step(self, a):
        pricetmp=[]
        actiontmp=[]
        trewardtmp=[]
        terminal=False
        action = self._action_set[a]

        self.fx_index = self.fx_index + 1
        fx_index = self.fx_index
        points = self.points

#action-plot
        actiontmp.append(fx_index)
        actiontmp.append(action+1+1)
        self.action_data.append(actiontmp)

#next price point
        data_index=fx_index+points-1
        if data_index>self.data_num-1:
            data_index=data_index%self.data_num
        self.price.append(self.data[data_index])

#price-plot
        pricetmp.append(fx_index)
        pricetmp.append(self.data[data_index])
        self.price_data.append(pricetmp)

#差价
        dprice = self.price[fx_index + points-1] - self.price[fx_index + points - 1-1]
        if action == -1:
            self.lastactiontime=0
            if self.hold_num <= 0:
                self.hold_num = self.hold_num + action
            if self.hold_num > 0:
                action = action * np.abs(self.hold_num)  ## number of open transction
                self.hold_num = 0  ###close all the open transcation
        if action == 1:
            self.lastactiontime=0
            if self.hold_num < 0:
                action = action * np.abs(self.hold_num)
                self.hold_num = 0
            if self.hold_num >= 0:
                self.hold_num = self.hold_num + action
        self.lastactiontime+=1 ###cumulate when action=0
#reward except cost
        rw = self.hold_num * dprice * self.lever - self.cost * np.abs(action) -self.swap*np.abs(self.hold_num)
        self.trw = self.trw + rw  ###total reward

#total reward plot
        trewardtmp.append(fx_index)
        trewardtmp.append(self.trw/self.max)
        self.treward_data.append(trewardtmp)

        #self.Account = self.Account - action * self.price[fx_index + points-1] - self.cost * np.abs(action)
        self.Account = self.Account + rw - self.cost * np.abs(action)

        sinTensor=self.price[fx_index:fx_index+points]
        sinTensor.append(self.hold_num*self.lever*0.0001/self.Account)   ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD
        #tmp = self.Account + self.hold_num * self.price[fx_index + points-1]
        sinTensor.append(self.Account/self.Account_All) ### use ratio instead of absolute value
        if self.Account < self.Account_All * (1 - self.lossRate):
            terminal = True
        sinTensor = np.asarray(sinTensor)
        return sinTensor.reshape(points+2,1), rw, terminal, {}
#return [sinTensor[:points].reshape(points,1),sinTensor[points:].reshape(2,1) ], rw, terminal, {}

    # return: (states, observations)
    def reset(self):
        self.hold_num=0
        self.Account=self.Account_All
        return self.step(0)

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