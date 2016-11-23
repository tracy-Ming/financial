--[[
Copyright (c) 2016 South China University of Technology .
Author：Qi Xiaoming
See LICENSE file for full terms of limited license.
]]

dqn={}
baseline={}

require 'torch'
require 'NeuralQLearner'
require 'Environment'

function dqn:myinfo()
  print("Started dqn initing...")
end

function baseline:myinfo()
  print("Started baseline initing...")
end

function baseline:baseInit(args)
  --print("Started baseline initing...")
  
  return self
end

function baseline:getSlop(screen,n)
      local x=(torch.range(1,n)*0.05):reshape(n,1)
      local xt=x:reshape(1,n)
      local y1=(screen[{{1,n},1}]):reshape(n,1)
       --print (x,y1)
       local x_average=torch.sum(x)/n
       local y_average=torch.sum(y1)/n
    --   print(x_average)
    --   print(y_average)
    --   print(torch.sum(xt*y1))
    --  print(x_average*y_average*n)
    --  print(torch.sum(xt*x))
    --  print(n*x_average*x_average)
      local b=(torch.sum(xt*y1)-x_average*y_average*n)/(torch.sum(xt*x)-n*x_average*x_average)
  
    return b
  end

function baseline:getAction(screen)
    --斜率涨跌
    --  local y1=(screen[{{1,7 },1}]):reshape(7,1)
    --  local y2=(screen[{{4,10 },1}]):reshape(7,1)
    --  print(getSlop(y2,7))
    -- print(getSlop(y1,7))  
    --  if(getSlop(y2,7)>getSlop(y1,7)) then 
    --    return 3
    --  end
    --  if(getSlop(y2,7)==getSlop(y1,7)) then 
    --    return 2
    --  end
    --  if(getSlop(y2,7)<getSlop(y1,7)) then 
    --    return 1
    --  end
  
      --斜率正负
      local y1=screen:sub(1,10)
      --print(getSlop(y1,10))  
      if(self:getSlop(y1,10)>0) then 
        return 3
      end
      if(self:getSlop(y1,10)==0) then 
        return 2
      end
      if(self:getSlop(y1,10)<0) then 
        return 1
      end
end

function torchSetup(_opt)
    _opt = _opt or {}
    local opt = table.copy(_opt)
    assert(opt)

    -- preprocess options:
    --- convert options strings to tables
    if opt.env_params then
        opt.env_params = str_to_table(opt.env_params)
    end
    if opt.agent_params then
        opt.agent_params = str_to_table(opt.agent_params)
        opt.agent_params.gpu       = opt.gpu
        opt.agent_params.best      = opt.best
        opt.agent_params.verbose   = opt.verbose
        if opt.network ~= '' then
            opt.agent_params.network = opt.network
        end
    end

    --- general setup
    opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(opt.tensorType)
    if not opt.threads then
        opt.threads = 4
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end

    math.random = nil
    opt.seed = opt.seed or 1
    torch.manualSeed(opt.seed)
    if opt.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if opt.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if opt.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    return opt
end

function setup(args)
      
      assert(args)
      args=torchSetup(args)
      --dqn:myinfo()
      baseline:myinfo()
      local data_env=env:Testing_init(args) --传参需要修改，规范
      local action=data_env:get_TestAction()
      --local agent=dqn["NeuralQLearner"](args)
      local agent=baseline:baseInit(args.agent_params)
      return data_env,action, agent,args
end

function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end
