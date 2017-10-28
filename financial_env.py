#coding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
#import tensorflow as tf
import MyAccount

class financialEnv(gym.Env):

    def __init__(self,opt,filepath,mode):#mode means training or testing
        self._opt=str_to_dic(opt)
        self.mode = mode
        args=self._opt
        self.account=MyAccount.MyAccount(args)    ###new add myAccount
        self.price_len = int(args['price_len']) or 100 #数据点
        self.fx_index = 0
        #self.hold_num = args['hold_num'] or 0
        #self.Account_All = args['Account_All'] or 3000
        #self.lossRate = args['lossRate'] or 0.6
        #self.Account = self.Account_All
        self.max = args['max'] or 100
        #minimum_order=1000 #### usually minimum order is 0.01 lot size =1000
        #self.lever = 500*minimum_order ####because action unit is 1, so I combine minmum_order with leverage
        #self.cost = 0.035
        #self.swap=0.01/60/24/self.lever*(500*1000)  ### each day interest rate is 0.01 for 0.01 lot and 500 leverage , and here it is the rate for each minute
        self.price = []
        self.trw = 0.
        self.rw=[]
        self.lastactiontime=0
        self.action_index = []
        self.maxdown=100
        self.maxdown_action=0
        self.mean = 0.0
        self.std = 0.0
        self.data,self.data_num=self.dataloading(filepath)
#plot
        self.price_data=[]#plot price
        self.action_data=[]#plot action
        self.treward_data=[]#plot total reward

#为price预存price_len个点
        for i in range(0,self.price_len):
            self.price.append(self.data[i])
            tmp = []
            tmp.append(i - self.price_len+1)
            tmp.append(self.data[i])
            self.price_data.append(tmp)

        #print self.price[0]
        #self._seed()
        self._action_set = [-2,-1,0,1,2]  ###define all possible action
        self.action_space = spaces.Discrete(len(self._action_set))
        '''
    #python2.7
    def dataloading(self,filepath):
        csvfile = np.loadtxt(filepath, dtype=np.str, delimiter=',')
        data = csvfile[0:, 2:].astype(np.float32)
         num=np.asarray(data).__len__()
        return np.asarray(data)[:,0],num
        '''
    def dataloading(self, filepath):
        csvfile = np.loadtxt(filepath, usecols=(1,2), dtype=np.float32, delimiter=',')
        data = csvfile[:,1]
        self.mean = np.mean(data)
        self.std = np.std(data)
        data = (data-self.mean)/self.std
        num = np.asarray(data).__len__()
        return np.array(data), num

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

        self.fx_index = self.fx_index + 1
        fx_index = self.fx_index
        price_len = self.price_len

#action-plot
        actiontmp.append(fx_index)
        actiontmp.append(action+1+1)
        self.action_data.append(actiontmp)

#next price point
        data_index=(fx_index+price_len-1)%self.data_num
        if data_index>=self.data_num-1:
            # data_index=data_index%self.data_num
            terminal = True
            #self.hold_num = 0
            '''
            self.account=MyAccount.MyAccount(self._opt)
            self.fx_index=0
            obs=self.price[0:price_len]
            obs.append(0)
            obs.append(0)
            obs.append(1)
            return np.asarray(obs).reshape(price_len+3,1),0,terminal,{}
           '''
        self.price.append(self.data[data_index])

#price-plot
        pricetmp.append(fx_index)
        pricetmp.append(self.data[data_index])
        self.price_data.append(pricetmp)

#差价
        dprice = (self.price[fx_index+price_len-1] - self.price[fx_index+price_len-1 -1]) * self.std
        env_info=self.getEnvData(self.price[fx_index+price_len-1])

        if action == -1:
            self.lastactiontime=0
            self.account.AddOrders(-1,env_info)
            # if self.hold_num <= 0:
            #     self.hold_num = self.hold_num + action
            # if self.hold_num > 0:
            #     action = action * np.abs(self.hold_num)  ## number of open transction
            #     self.hold_num = 0  ###close all the open transcation
        if action == 1:
            self.lastactiontime=0
            self.account.AddOrders(1,env_info)
            # if self.hold_num < 0:
            #     action = action * np.abs(self.hold_num)
            #     self.hold_num = 0
            # if self.hold_num >= 0:
            #     self.hold_num = self.hold_num + action
        self.lastactiontime+=1 ###cumulate when action=0
        if action == -2:
            self.lastactiontime = 0
            self.account.CloseBidOrders()
        if action == 2:
            self.lastactiontime = 0
            self.account.CloseAskOrders()

#reward except cost
        #rw = self.hold_num * dprice * self.lever - self.cost * np.abs(action) -self.swap*np.abs(self.hold_num)
        rw = self.account.ComputeProfit(dprice,action)
        self.trw = self.trw + rw  ###total reward

#total reward plot
        trewardtmp.append(fx_index)
        trewardtmp.append(self.trw/self.max)
        self.treward_data.append(trewardtmp)

        #self.Account = self.Account - action * self.price[fx_index + price_len-1] - self.cost * np.abs(action)
        #self.Account = self.Account + rw - self.cost * np.abs(action)

        sinTensor=self.price[fx_index:fx_index+price_len]
        #sinTensor.append(self.hold_num*self.lever*0.0001/self.account.Account)   ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD
        sinTensor.append(self.account.getAskLots()) ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD     Askorders
        sinTensor.append(self.account.getBidLots()) ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD     Bidorders
        #tmp = self.Account + self.hold_num * self.price[fx_index + price_len-1]
        sinTensor.append(self.account.Account/self.account.Account_All) ### use ratio instead of absolute value
        if self.account.Account < self.account.Account_All * (1 - self.account.lossRate) and self.mode == "train":
            terminal = True
        sinTensor = np.asarray(sinTensor)
        return sinTensor.reshape(price_len+3,1), rw, terminal, {}
#return [sinTensor[:price_len].reshape(price_len,1),sinTensor[price_len:].reshape(2,1) ], rw, terminal, {}

    # return: (states, observations)
    def reset(self):
        self.account = MyAccount.MyAccount(self._opt)
        self.fx_index = 0
        self.price = []
        self.trw = 0.
        self.rw = []
        self.lastactiontime = 0
        self.action_index = []
        self.price_data = []  # plot price
        self.action_data = []  # plot action
        self.treward_data = []  # plot total reward

        # 为price预存price_len个点
        for i in range(0, self.price_len):
            self.price.append(self.data[i])
            tmp = []
            tmp.append(i - self.price_len + 1)
            tmp.append(self.data[i])
            self.price_data.append(tmp)
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
