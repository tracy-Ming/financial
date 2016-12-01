env={}

require 'torch'
require 'data_load'
require 'network_config'
require 'gnuplot'

function env:init(args)
        
        setDL_network(args)

end

function env:setDL_network(args)
    local nel=cnn_config(args)
    
    return nel
end

function env:get_TestAction()
    return {-1,0,1}
end

function env:get_TestData(sin_index, dt)

         local x=torch.uniform(0,self.noise) 
         local y=math.pow(-1,torch.random(1,100))
    --print(math.abs(math.sin(sin_index*dt+0.001)+1+self.noise+x*y ))
    return math.abs(math.sin(sin_index*dt+0.001)+1+self.noise+x*y )
        
end

function env:Testing_init(opt)
      local args=opt.env_params
      self.opt=opt
      --info about data used
      self.points=args.points or 10             --the num of dataset 
      self.dt=args.dt or 0.05                        --
      self.sin_index=args.sin_index or 0     --
      self.hold_num=args. hold_num or 0  --dollar
      self.noise=args.noise or 0                   --noise
      --info about my account and lossrate
      self.Account_All=args.Account_All or 100
      self.lossRate=args.lossRate or 0.6      --if lose 40% stop
      self.Account=self.Account_All             --RMB
      self.max=args.max or 100
      self.lever=1
      self.cost=0
      self.price={}
      self.sindex={}
      self.shb={}
      self.trw=0
      self.own={} 
      self.action_index={}
        
        for i=1,self.points do 
          self.price[i]=self:get_TestData(i-self.points,self.dt)
          self.sindex[i]=i-self.points
        end
      return self
end

function env:NewTestState()
      --print("New_State.....")
      self.hold_num=0
      self.Account=self.Account_All
      local state,reward,terminal = self:TestStep(0)
      
      return  state,reward,terminal
end

function env:AnotherTestState()
      --print("New_State.....")
      local state,reward,terminal = self:TestStep(0)
      
      return  state,reward,terminal
end

function env:TestStep(action)
 
      self.sin_index = self.sin_index + 1
      local sin_index=self.sin_index
      local points=self.points
      local dt=self.dt
      self.shb[sin_index]=action+1+1-----plot--noise +6 --better in picture 
      self.sindex[sin_index+points]=sin_index
      self.action_index[sin_index]=sin_index
    
      --next time price
      self.price[sin_index+points]=self:get_TestData(sin_index,dt)
   
      local terminal =  false
      local dprice  = self.price[sin_index+points]-self.price[sin_index+points-1]
        
        if action==-1 then 
           if self.hold_num<=0 then
               self.hold_num=self.hold_num+action
           end
           if self.hold_num>0 then
               action=action*math.abs(self.hold_num)
               self.hold_num=0             
             end
        end
        if action==1 then 
            if self.hold_num<0 then
                action=action*math.abs(self.hold_num)
                self.hold_num=0
           end
            if self.hold_num>=0 then
                self.hold_num=self.hold_num+action
             end
        end
   
          -------------------print___info------------------------
--        print (sin_index+points , self.price[sin_index+points] )
--        print (sin_index+points-1 , self.price[sin_index+points-1]  ) 
--        print ("reward=",self.hold_num,"X",dprice," = ",self.hold_num*dprice)
   
        --hold_num  = hold_num  + action 
          local rw=self.hold_num  * dprice *self.lever -self.cost * math.abs(action) ---action
          self.trw=self.trw+rw
          self.own[sin_index]=self.trw/self.max
          
--            print("before action",self.Account)
            --hold_num  = hold_num  + action --buy/hold/sell 1$ at point 12
            self.Account  = self.Account  - action  * self.price[sin_index+points] -self.cost * math.abs(action)
--            print("after -",action,"X",self.price[sin_index+points],-action  * self.price[sin_index+points]," = ",self.Account)

   
          local sinTensor = torch.Tensor(points+2,1):fill(0.01)
          sinTensor[{{1,points},1}]=torch.Tensor(self.price):reshape(#self.price,1):sub(sin_index+1,sin_index+points)
          
            sinTensor[points+1]  = self.hold_num
            local tmp=self.Account  + self.hold_num  * self.price[sin_index+points]
            sinTensor[points+2]  = tmp
            
--            print(tmp)
--            print(self.Account_All * (1-self.lossRate))
            if tmp <  self.Account_All * (1-self.lossRate) then
                terminal = true
            end
          return sinTensor, rw, terminal
end

function env:draw()
   local path='test'
   path=path..#self.action_index..'_noise_'..self.noise
    gnuplot.pngfigure('src/result/'..path..'.png')
    gnuplot.plot({torch.Tensor(self.sindex), torch.Tensor(self.price)},{torch.Tensor(self.action_index), torch.Tensor(self.shb)} , {torch.Tensor(self.action_index),torch.Tensor(self.own)})
--    gnuplot.plot({torch.Tensor(self.sindex), torch.Tensor(self.price)}, {torch.Tensor(self.action_index),torch.Tensor(self.own)})
    gnuplot.plotflush()
end

function env:get_FXAction()
        return {-1,0,1}
end

function env:FX_init(opt)
      local args=opt.env_params
      self.opt=opt
      --info about data used
      self.points=args.points or 10             --the num of dataset 
      self.fx_index=0     --
      self.hold_num=args. hold_num or 0  --dollar
      --info about my account and lossrate
      self.Account_All=args.Account_All or 100
      self.lossRate=args.lossRate or 0.6      --if lose 40% stop
      self.Account=self.Account_All             --RMB
      self.max=args.max or 100
      self.lever=1
      self.cost=0
      self.price={}
      self.sindex={}
      self.shb={}
      self.trw=0
      self.own={} 
      self.rw={}
      self.action_index={}
      self.maxdown=100
      self.maxdown_action=0
      self.epochs=0
      local data,num=load:data_loading(opt)
      self.data=data:select(2,3)
      self.data_num=num
      
      for i=1,self.points do 
          self.price[i]=self.data[i]
          self.sindex[i]=i
        end
      
      return self
end

function env:FX_Step(action)
      self.fx_index = self.fx_index + 1
      local fx_index=self.fx_index
      local points=self.points
      
      self.shb[fx_index]=action+1+1-----plot
      self.sindex[fx_index+points]=fx_index
      self.action_index[fx_index]=fx_index
     
     local data_index=fx_index+points
     if env:shutdown() then --循环训练数据
         data_index=data_index%self.data_num+1
      end
    
      --next time price
      self.price[fx_index+points]=self.data[data_index]
      if (fx_index+points)%10000==0  then print(fx_index+points)end
      local terminal =  false
      local dprice  = self.price[fx_index+points]-self.price[fx_index+points-1]
        
        if action==-1 then 
           if self.hold_num<=0 then
               self.hold_num=self.hold_num+action
           end
           if self.hold_num>0 then
               action=action*math.abs(self.hold_num)
               self.hold_num=0             
             end
        end
        if action==1 then 
            if self.hold_num<0 then
                action=action*math.abs(self.hold_num)
                self.hold_num=0
           end
            if self.hold_num>=0 then
                self.hold_num=self.hold_num+action
             end
        end
   
          -------------------print___info------------------------
--        print (sin_index+points , self.price[sin_index+points] )
--        print (sin_index+points-1 , self.price[sin_index+points-1]  ) 
--        print ("reward=",self.hold_num,"X",dprice," = ",self.hold_num*dprice)
   
        --hold_num  = hold_num  + action 
          local rw=self.hold_num  * dprice *self.lever -self.cost * math.abs(action)   ---action
          self.trw=self.trw+rw
          self.own[fx_index]=self.trw/self.max
          self.rw[fx_index]=rw
          if  fx_index>=2 and rw-self.rw[fx_index-1] <=self.maxdown  then
              self.maxdown=rw-self.rw[fx_index-1]
              self.maxdown_action= self.shb[fx_index-1]
           end
--            print("before action",self.Account)
            --hold_num  = hold_num  + action --buy/hold/sell 1$ at point 12
            self.Account  = self.Account  - action  * self.price[fx_index+points] -self.cost * math.abs(action)
--            print("after -",action,"X",self.price[sin_index+points],-action  * self.price[sin_index+points]," = ",self.Account)

   
          local Tensor = torch.Tensor(points+2,1):fill(0.01)
          Tensor[{{1,points},1}]=torch.Tensor(self.price):sub(fx_index+1,fx_index+points)
          
            Tensor[points+1]  = self.hold_num
            local tmp=self.Account  + self.hold_num  * self.price[fx_index+points]
            Tensor[points+2]  = tmp
            
--            print(tmp)
--            print(self.Account_All * (1-self.lossRate))
            if tmp <  self.Account_All * (1-self.lossRate) then
                terminal = true
            end
          return Tensor, rw, terminal
end

function env:New_FXState()
      self.hold_num=0
      self.Account=self.Account_All
      local state,reward,terminal = self:FX_Step(0)
      
      return  state,reward,terminal
end

function env:AnotherFXState()
      local state,reward,terminal = self:FX_Step(0)
      
      return  state,reward,terminal
end

function env:Step(action)
      if self.opt.env=='sin_data' then
             return self:TestStep(action)
      else
             return self:FX_Step(action)
      end
end

function env:getState()
      if self.opt.env=='sin_data' then
             return self:NewTestState()
      else
             return self:New_FXState()
      end
end


function env:newState()
      if self.opt.env=='sin_data' then
             return self:AnotherTestState()
      else
             return self:AnotherFXState()
      end
end

function env:shutdown()
      if #self.price>=self.data_num then 
         return true
      else
         return false
      end 
end