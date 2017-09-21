#coding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
#import tensorflow as tf
import MyAccount


class simulationEnv(gym.Env):

    def __init__(self,opt,mode):
        self._opt = str_to_dic(opt)
        self.mode = mode
        args = self._opt
        self.account = MyAccount.MyAccount(args)  ###new add myAccount
        self.price_len = int(args['price_len']) or 100  # 数据点
        self.sin_index = args['sin_index'] or 0
        self.dt = args['dt'] or 0.05
        self.noise = args['noise'] or 0
        self.max = args['max'] or 100

        self.price = []
        self.trw = 0
        self.rw=[]
        self.lastactiontime = 0
        self.action_index = []
        self.maxdown = 100
        self.maxdown_action = 0

        self.price_data = []  # plot price
        self.action_data = []  # plot action
        self.treward_data = []  # plot total reward

#为price预存points个点
        for i in xrange(0,self.price_len):
            tmpP=self.get_TestData(i - self.price_len+1, self.dt)
            self.price.append(tmpP)
            tmp = []
            tmp.append(i - self.price_len + 1)
            tmp.append(tmpP)
            self.price_data.append(tmp)

        #self._seed()
        self._action_set = [-2,-1,0,1,2]
        self.action_space = spaces.Discrete(len(self._action_set))


    def get_TestData(self,sin_index,dt):
        x=np.random.uniform(0,self.noise)
        y=np.power(-1,np.random.random_integers(1,100))
        # return np.sin(sin_index * dt + 0.001) * (sin_index % 600) / 300.0 + 2.0 + self.noise + x * y
        return np.abs(np.sin(sin_index*dt+0.001) + 1 + self.noise + x * y)

    def getEnvData(self,price):
        opt={}
        #opt["id"]=
        opt["time"] = ""
        opt["lots"] = 0.01
        opt["symbol"] = "eurusd"
        opt["price"] = price
        opt["stoploss"] = 0.0
        opt["takeprofit"] = 0.0
        opt["closeprice"] = 0.0
        opt["cost"] = 0.0
        opt["swap"] = 0.0
        opt["profit"] = 0.0
        return opt

    def step(self, a):
        pricetmp=[]
        actiontmp=[]
        trewardtmp=[]
        terminal=False
        action = self._action_set[a]

        self.sin_index = self.sin_index + 1
        sin_index = self.sin_index
        price_len = self.price_len
        dt = self.dt

# action-plot
        actiontmp.append(sin_index)
        actiontmp.append(action + 1 + 1)
        self.action_data.append(actiontmp)

#next price point
        tmpP=self.get_TestData(sin_index, dt)
        self.price.append(tmpP)

# price-plot
        pricetmp.append(sin_index)
        pricetmp.append(tmpP)
        self.price_data.append(pricetmp)

#差价
        data_index = sin_index + price_len - 1
        dprice = self.price[data_index] - self.price[data_index-1]
        env_info = self.getEnvData(self.price[data_index])

        rw = 0.0
        if action == 0:
            rw = self.account.ComputeProfit(dprice,action)
            self.lastactiontime += 1  ###cumulate when action=0
        if action == -1:
            #self.account.Account -=  env_info["lots"] * self.account.standard_order * self.price[data_index]
            self.lastactiontime=0
            self.account.AddOrders(-1,env_info)
            rw = self.account.ComputeProfit(dprice, action)
            #action * env_info["lots"] * self.account.standard_order * dprice * self.account.lever
            # if self.hold_num <= 0:
            #     self.hold_num = self.hold_num + action
            # if self.hold_num > 0:
            #     action = action * np.abs(self.hold_num)  ## number of open transction
            #     self.hold_num = 0  ###close all the open transcation
        if action == 1:
            #self.account.Account -= env_info["lots"] * self.account.standard_order * self.price[data_index]
            self.lastactiontime=0
            self.account.AddOrders(1,env_info)
            rw = self.account.ComputeProfit(dprice,action)
            #action * env_info["lots"] * self.account.standard_order * dprice * self.account.lever
            # if self.hold_num < 0:
            #     action = action * np.abs(self.hold_num)
            #     self.hold_num = 0
            # if self.hold_num >= 0:
            #     self.hold_num = self.hold_num + action
        if action == -2:
            self.lastactiontime = 0
            self.account.CloseBidOrders()
            rw = self.account.ComputeProfit(dprice,action)
            # if len(self.account.BidOrders)==0:
            #     rw = -1000
            #rw = rw + future_rw
        if action == 2:
            self.lastactiontime = 0
            self.account.CloseAskOrders()
            rw = self.account.ComputeProfit(dprice, action)
            # if len(self.account.AskOrders)==0:
            #     rw = -1000
            #rw = rw + future_rw

        #reward except cost
        #rw = self.hold_num * dprice * self.lever - self.cost * np.abs(action) -self.swap*np.abs(self.hold_num)

        self.trw = self.trw + rw  ###total reward5

# total reward plot
        trewardtmp.append(sin_index)
        trewardtmp.append(self.trw / self.max)
        self.treward_data.append(trewardtmp)

        sinTensor = self.price[sin_index:sin_index + price_len]
        sinTensor.append(self.account.getAskLots())  ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD     Askorders
        sinTensor.append(self.account.getBidLots())
        sinTensor.append(self.account.Account / self.account.Account_All)  ### use ratio instead of absolute value
        # if self.buy and action == 1:
        #     terminal = True
        # if self.sell and action == -1:
        #     terminal = True
        if self.account.Account < self.account.Account_All * (1 - self.account.lossRate) and self.mode == "train":
            terminal = True
        sinTensor = np.asarray(sinTensor)
        return sinTensor.reshape(price_len + 3, 1), rw, terminal, {}

    # return: (states, observations)
    def reset(self):
        self.account = MyAccount.MyAccount(self._opt)
        return self.step(2)

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
