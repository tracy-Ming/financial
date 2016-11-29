--[[
Copyright (c) 2016 South China University of Technology .
Author：Qi Xiaoming
See LICENSE file for full terms of limited license.
]]

require 'torch'
require 'initenv'
require 'sys'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')
cmd:option('-env', 'sin_data', 'name of environment to use')
cmd:option('-env_params', 'points=10,dt=0.05,sin_index=0,noise=0,hold_num=0,Account_All=3000,lossRate=0.6,max=500', 'string of environment parameters')
cmd:option('-filepath', 'EURUSD60_train.csv', 'FX_data used to')
--cmd:option('-env_params', 'ep_endt=1000000,discount=0.99,learn_start=50000', 'string of environment parameters')
--cmd:option('-pool_frms', '','string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-name', 'dqn_financial', 'name of the model')
cmd:option('-network', 'trainning_dqn', 'load pretrained network')
cmd:option('-agent', 'NeuralQLearner', 'name of agent file to use')
cmd:option('-agent_params', 'lr=0.00025,ep=1,ep_end=0.1,ep_endt=1000000,discount=0.99,hist_len=1,learn_start=50,replay_memory=1000000,update_freq=4,n_replay=1,'..
                   'network=\'convnet_atari3\',preproc=\"net_downsample_2x_full_y\",state_dim=12,'..
                   'minibatch_size=32,rescale_r=1,bufferSize=45,valid_size=30,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,'saves the agent network in a separate file')
cmd:option('-prog_freq', 100000, 'frequency of progress output')
cmd:option('-save_freq', 125, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 250, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')
cmd:option('-steps',600000, 'the size of the testset')
cmd:option('-eval_steps', 125, 'number of evaluation steps')
cmd:option('-verbose', 2,'the higher the level, the more information is printed to screen')
cmd:option('-threads', 2000, 'number of BLAS threads')
cmd:option('-gpu', 0, 'gpu flag')

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
-- print(opt)

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward
p_reward=0
n_reward=0
T_reward=0
-- start a new game
local state, reward, terminal = data_env:getState()
 print( "first state: ")
 print(state )

print("Iteration ..", step)
--local win = nil
while step < opt.steps do
    
--    if opt.env ~='sin_data' then
--       if env:shutdown() then
--         print("no enough data...")
--         break
--       end
--    end

    step = step + 1
    local action_index = agent:perceive(reward, state, terminal)

    print ("opt.itera: ",step)
--    print("action: ",shb_actions[action_index])

    if not terminal then
        state, reward, terminal = env:Step(shb_actions[action_index], true)
        
--        print ("reward: ",reward)
--        print( "next state: ")
--        print(state) 
   
    else
          state, reward, terminal = env:newState()
    end
   
    -- display screen
    --win = image.display({image=screen, win=win})

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then

        state, reward, terminal = env:getState()
        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action_index = agent:perceive(reward, state, terminal, true, 0.05)
--print('----',action_index,'----')
            -- in test mode (episodes don't end when losing a life)
        state, reward, terminal = env:Step(shb_actions[action_index])

            -- display screen
            -- win = image.display({image=screen, win=win})

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end
            
             if reward>0 then
                     p_reward=p_reward+1
             end
             if reward<0 then
                     n_reward=n_reward+1
             end
              T_reward=T_reward+reward

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                break
               -- state, reward, terminal = env:newState()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
     if step%1000 == 0 then
            print("the num of +/- rewards is ", p_reward,n_reward,p_reward/(p_reward+n_reward))
            print("average reward is ",T_reward/(p_reward+n_reward)) end
end
        print("Finished traning, close window to exit!")