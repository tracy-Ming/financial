env={}

require 'torch'
require 'data_load'
require 'network_config'
require 'gnuplot'

local data=env["data_load"]()

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

function env:Testing_init(args)

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
          local rw=self.hold_num  * dprice---action
          self.trw=self.trw+rw
          self.own[sin_index]=self.trw/self.max
          
--            print("before action",self.Account)
            --hold_num  = hold_num  + action --buy/hold/sell 1$ at point 12
            self.Account  = self.Account  - action  * self.price[sin_index+points]
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
    gnuplot.plotflush()
end

function env:get_FXAction()

end

function env:get_FXData()

end

function env:get_FXState()

end
