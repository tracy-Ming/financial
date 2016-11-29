--[[
Copyright (c) 2016 South China University of Technology .
Author：Qi Xiaoming
See LICENSE file for full terms of limited license.
]]

require 'torch'
require 'initenv'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Test Agent in Environment:')
cmd:text()
cmd:text('Options:')
cmd:option('-loop', 500, 'the size of the testset')
cmd:option('-env', 'sin_data', 'name of environment to use')
cmd:option('-env_params', 'points=10,dt=0.05,sin_index=0,noise=0,hold_num=0,Account_All=3000,lossRate=0.6,max=100', 'string of environment parameters')
cmd:option('-filepath', 'EURUSD60_test.csv', 'FX_data used to')
cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', 'dqn_financial.t7', 'reload pretrained network')
cmd:option('-agent', 'NeuralQLearner', 'name of agent file to use')
cmd:option('-agent_params', 'lr=0.00025,ep=1,ep_end=0.1,ep_endt=1000000,discount=0.99,hist_len=1,learn_start=50,replay_memory=1000000,update_freq=4,n_replay=1,'..
                   'network=\'convnet_atari3\',preproc=\"net_downsample_2x_full_y\",state_dim=12,'..
                   'minibatch_size=32,rescale_r=1,bufferSize=45,valid_size=30,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-verbose', 2,'the higher the level, the more information is printed to screen')
cmd:option('-threads', 2000, 'number of BLAS threads')
cmd:option('-gpu', 0, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

cmd:text()
local opt = cmd:parse(arg)

-- General setup.
local data_env,shb_actions,agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- start a new game
local state, reward, terminal = data_env:getState()

--print("Started ...")
N_reward=0
p_reward=0
n_reward=0
T_reward=0
--
-- start playing
   for i=2,opt.loop do
     --print(terminal)
     if  terminal then
       break
       end
       
     if opt.env ~='sin_data' then
       if env:shutdown() then
         print("ending...")
         break
       end
    end
       
   -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    -- choose the best action
    --print("Loop----------------------",i)
    local action_index = agent:perceive(reward, state, terminal, true, 0.05)
    --print("next action 1/2/3 for S/H/B",action_index)
    
    state, reward, terminal = env:Step(shb_actions[action_index])
   
   T_reward=reward+T_reward
   
   if(reward~=0) then 
      N_reward=N_reward+1
    end
   
   if(reward>0) then 
      p_reward=p_reward+1
    end
    if reward<0 then
      n_reward=n_reward+1
    end
   n_reward=N_reward-p_reward

end
    print("the num of +/- rewards is ", p_reward,n_reward,p_reward/(p_reward+n_reward))
    print("total reward is ",T_reward)
    print("average reward is ",T_reward/(p_reward+n_reward))
    data_env:draw()
    print("Finished testing, close window to exit!")