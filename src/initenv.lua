--[[
Copyright (c) 2016 South China University of Technology .
Author：Qi Xiaoming
See LICENSE file for full terms of limited license.
]]

dqn = {}

require 'torch'
require 'Environment'
require 'NeuralQLearner'
require 'TransitionTable'
require 'nnutils'

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


function setup(_opt)
    assert(_opt)

    --preprocess options:
    --- convert options strings to tables
    _opt.env_params = str_to_table(_opt.env_params)
    _opt.agent_params = str_to_table(_opt.agent_params)
    if _opt.agent_params.transition_params then
        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end

    --- first things first
    local opt = torchSetup(_opt)
    
    local data_env=env:Testing_init(opt) 
    local action=data_env:get_TestAction()
    if opt.env ~='sin_data' then
        data_env=env:FX_init(opt) 
        action=data_env:get_FXAction()
    end
    -- agent options
    _opt.agent_params.actions   = action
    _opt.agent_params.gpu       = _opt.gpu
    _opt.agent_params.best      = _opt.best
    if _opt.network ~= '' then
        _opt.agent_params.network = _opt.network
    end
    _opt.agent_params.verbose = _opt.verbose

    local agent = dqn[_opt.agent](_opt.agent_params)

    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
    end

    return data_env, action, agent, opt
end



--- other functions

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