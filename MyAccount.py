# coding=utf-8
import numpy as np
class MyAccount:
    def __init__(self,args):
        #self.AccountName=args["AccountID"] or "ming xiao"
        self.Account_All=args['Account_All'] or 3000
        self.pip=0.0001
        self.Account=self.Account_All
        self.AskOrders=[]
        self.BidOrders=[]
        self.lossRate = args['lossRate'] or 0.6
        self.standard_order=100000 #### 1 lot size=100 000
        minimum_order = 1000  #### usually minimum order is 0.01 lot size =1000
        self.lever = 500
        self.cost = 0#0.035
        self.swap = 0#0.01 / 60 / 24 / self.lever * (500 * 1000)  ### each day interest rate is 0.01 for 0.01 lot and 500..
                                                                #  leverage , and here it is the rate for each minute

    def getMt4Orders(self,Order):
        opt={}
        opt["id"]=Order["ticket"]
        opt["time"] = Order["opentime"]
        opt["lots"] = Order["lots"]
        opt["symbol"] = Order["symbol"]
        opt["price"] = Order["openprice"]
        opt["stoploss"] = Order["stoploss"]
        opt["takeprofit"] = Order["takeprofit"]
        opt["closeprice"] = Order["closeprice"]
        opt["cost"] = Order["commission"]
        opt["swap"] = Order["swap"]
        opt["profit"] = Order["profit"]
        return opt

    def SetMyOrders(self,Orders):
        for order in Orders:
            opt = self.getMt4Orders(order)
            if order["type"]== 0:
                self.AskOrders.append(opt)
            if order["type"]== 1:
                self.BidOrders.append(opt)

    def AddOrders(self,direction,order):
        #print "Adding order is ",order["price"]
        if direction== 1:
            self.AskOrders.append(order)
        if direction== -1:
            self.BidOrders.append(order)

    def CloseAskOrders(self):
        rw = 0.0
        for ask_order in self.AskOrders:
            # ask_order["profit"] + dprice  means each askorder's total profits
            rw = rw + ask_order["lots"] * self.standard_order * ( ask_order["profit"]  ) * self.lever - self.cost
        self.Account = self.Account + rw
        self.AskOrders = []
        return rw

    def CloseBidOrders(self):
        rw = 0.0
        for bid_order in self.BidOrders:
            rw = rw - bid_order["lots"] * self.standard_order * ( bid_order["profit"]  ) * self.lever - self.cost
        self.Account = self.Account + rw
        self.BidOrders = []
        return rw

    def ComputeProfit(self,dprice,action):
        rw=0.0  ## reward between time t and t+1 for all long/short orders
        for ask_order in self.AskOrders:
            # ask_order["cost"] = self.cost
            # ask_order["swap"] = self.swap * ask_order["lots"]/0.01
            ask_order["profit"] += dprice
            rw = rw + ask_order["lots"] * self.standard_order * dprice * self.lever
        for bid_order in self.BidOrders:
            bid_order["profit"] += dprice
            rw = rw - bid_order["lots"] * self.standard_order * dprice * self.lever
        rw = rw - self.cost * (np.abs(action)%2!=0)
        if action == 1:
            rw = rw - self.swap * self.AskOrders[-1]["lots"] / 0.01
        if action == -1:
            rw = rw - self.swap * self.BidOrders[-1]["lots"] / 0.01
        self.Account = self.Account + rw
        return rw

    ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD
    def getAskLots(self):
        lots=0.0
        for ask_order in self.AskOrders:
            lots += ask_order["lots"]
        return lots * self.standard_order * self.pip *self.lever/self.Account

    ### scale to percentage of balance in use , 0.0001 is 1 pip for EURUSD
    def getBidLots(self):
        lots=0.0
        for bid_order in self.BidOrders:
            lots += bid_order["lots"]
        return lots * self.standard_order * self.pip *self.lever/self.Account

# args={}
# args["AccountID"]="ming xiao"
# args["Account_All"]=3000
# args["lossRate"]=0.6
# acc=MyAccount(args)
# opt={}
# opt["id"]="35954030"
# opt["time"]="2017.0405 14:55:27"
# opt["lots"]=0.01
# opt["pair"]="EURUSD"
# opt["price"]=1.06801
# opt["stoploss"]=0.0
# opt["takeprofit"]=0.0
# opt["price2"]=1.05905
# acc.AddOrders(1,opt)
# acc.AddOrders(1,opt)
# import json
# file=open("orders.txt",'w')
# for i in acc.AskOrders:
#     print json.dumps(i)
#     file.write(json.dumps(i))
#     file.write('\n')
# file.close()
# print "************"
# print acc.AskOrders
# acc=MyAccount(args)
# print "-----------"
# print acc.AskOrders

