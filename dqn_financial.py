# coding=utf-8
# 1060,done, took 2729.434 seconds
# k-80,done, took 11504.376 seconds
from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, LSTM, Reshape, Merge, Dropout, Highway,Concatenate,Input,Lambda
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TrainIntervalLogger
from keras.callbacks import Callback
import financial_env
import financial_env_for_simulation
from keras.layers.convolutional_recurrent import ConvLSTM2D

price_len = 20
add_another3D =3
WINDOW_LENGTH = 1


class financialProcessor(Processor):
    def process_observation(self, observation):
        # print observation
        if np.asarray(observation).shape[0] == 4:
            return observation[0]#[observa  tion[0][:(len(observation[0])-3)] ,observation[0][-3:] ]
        else:
            return observation#[observation[:(len(observation)-3)],observation[-3:]]

    def process_reward(self, reward):
        return reward#np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='real')
parser.add_argument('--training-path', type=str, default='dataset/EURUSD60_train_1200.csv')
parser.add_argument('--training-steps', type=int, default=600000)
parser.add_argument('--testing-path', type=str, default='dataset/EURUSD60_train_1200.csv')#default='EURUSD60_train_12min.csv')
parser.add_argument('--testing-steps', type=str, default=1200)
parser.add_argument('--env-params', type=str, default='price_len=' + str(price_len) + ',dt=0.05,sin_index=0,noise=0,hold_num=0,Account_All=3000,lossRate=0.6,max=40000')
parser.add_argument('--weights', type=str, default='real')
parser.add_argument('--update', type=str, default='n')#update the model when update=y


args = parser.parse_args()

datafile = args.training_path
if args.mode=='test':
    datafile=args.testing_path

# Get the environment and extract the number of actions.
if args.env_name != 'real':
    env = financial_env_for_simulation.simulationEnv(args.env_params,args.mode)
else:
    env = financial_env.financialEnv(args.env_params, datafile,args.mode)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# We patch the environment to be closer to what Mnih et al. actually do: The environment
# repeats the action 4 times and a game is considered to be over during training as soon as a live
# is lost.
# def _step(a):
#     reward = 0.0
#     action = env._action_set[a]
#     lives_before = env.ale.lives()
#     for _ in range(4):
#         reward += env.ale.act(action)
#     ob = env._get_obs()
#     done = env.ale.game_over() or (args.mode == 'train' and lives_before != env.ale.lives())
#     return ob, reward, done, {}
# env._step = _step

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).


dropout_rate = 0.5
num_layers = 3

useMerge = False
useLSTM = False
if not useMerge:
    model = Sequential()
    # input_data size (20+2)*1
    INPUT_SHAPE = (add_another3D + price_len, 1)  ##need to have some way to normalize the last 2 dimension
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    if useLSTM:
        model.add(Reshape(INPUT_SHAPE,input_shape=input_shape))
        model.add(LSTM(10))
        model.add(Dropout(dropout_rate))
        # model.add(Dense(5))
        # model.add(Flatten())
        # for index in range(num_layers):
        #     model.add(Highway(activation='relu'))
        #     model.add(Dropout(dropout_rate))
    else:  # CNN
        model.add(Conv2D(32, (8, 1), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 1), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(51))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    # model.add(Activation('softmax'))
    model.add(Activation('linear'))
    print(model.summary())

else:
    ####use Lambda Layer to slice Tensor (23,1) into (20,1) (3,)
    def slice(x,index,up):
        if up:
            return x[:,:index,]
        else:
            return x[:,index:,]

    Obs_input = Input(shape=(1,price_len+add_another3D , 1) )
    Obs = Reshape((price_len+add_another3D,1))(Obs_input)
    price_input = Lambda(slice,output_shape=(price_len,1),arguments={"index":price_len,"up":True})(Obs)
    account_input = Lambda(slice,output_shape=(add_another3D,1),arguments={"index":price_len,"up":False})(Obs)
    lstm_out = LSTM(15,return_sequences=True)(price_input)
    lstm_out = LSTM(10)(lstm_out)
    account_out = Reshape((3,))(account_input)
    account_out = Dense(6)(account_out)
    x = keras.layers.concatenate([lstm_out, account_out])
    # We stack a deep densely-connected network on top
    x = Dense(64,activation='relu')(x)
    x = Dense(nb_actions, activation='softmax')(x)
    model= Model(inputs=Obs_input, outputs=x)
    print model.summary()



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = financialProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!


dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, delta_range=(-1., 1.),
               target_model_update=10000, train_interval=4)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

class eps_History(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        self.reward_list=[]
        self.action_list=[]
        self.price=env.price
        self.reset()

    def reset(self):
        self.maxdown=10000
        self.maxdown_action=0
        self.p_reward=0
        self.n_reward=0
        self.T_reward=0.0
        self.episode_rewards = []

    def on_step_begin(self, step, logs):
        if self.step % self.interval == 0:
            # print "*-*----*-*-*-*-*-*-*-*-*-*-"
            # print len(self.episode_rewards)
            #if len(self.episode_rewards) > 0:
                print ("+/- rewards are",self.p_reward," / ",self.n_reward," / ",self.p_reward/(self.p_reward+self.n_reward+0.00001))
                print ("total reward is ",self.T_reward)
                print ("average reward is ",self.T_reward/self.interval)
                print ("maxdown is ",self.maxdown)
                print ("maxdown action is ",self.maxdown_action)
                # print self.price.__len__()
                print ('')
                self.reset()


    def on_step_end(self, step, logs):
        self.step += 1
        self.reward_list.append(logs['reward'])
        self.action_list.append(logs['action'])
        if logs['reward']>0:
            self.p_reward=self.p_reward+1
        if logs['reward']<0:
            self.n_reward=self.n_reward+1
        self.T_reward=self.T_reward+logs['reward']
        if self.step>2:
            if self.reward_list[step]-self.reward_list[step-1]<self.maxdown:
                self.maxdown=self.reward_list[step]-self.reward_list[step-1]
                self.maxdown_action=self.action_list[step]

    def on_episode_end(self, episode, logs):
        self.episode_rewards.append(logs['episode_reward'])

suffix = args.training_path.split('/')[-1].split('.')[0]
if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'model/dqn_{}_weights_{}.h5f'.format(args.env_name,suffix)
    checkpoint_weights_filename = 'model/dqn_' + args.env_name + '_weights_{step}_'+ suffix +'.h5f'
    log_filename = 'model/dqn_{}_log_{}.json'.format(args.env_name,suffix)
    res=eps_History(interval=60000)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=60000*5)]
    callbacks += [FileLogger(log_filename, interval=100)]
    callbacks += [res]

    if args.update == 'y':
        if args.weights:
            weights =  'model/dqn_{}_weights_{}.h5f'.format(args.weights,suffix)
            print (weights)
            model.load_weights(weights)
            print "loaded weight file:",weights_filename
    dqn.fit(env, callbacks=callbacks, nb_steps=args.training_steps, log_interval=60000)
    ins = np.array(dqn.ins_info).reshape([-1,price_len+add_another3D])
    q = np.array(dqn.q_info).reshape([-1,5])
    ins_q = np.hstack((ins,q))

    np.savetxt("intermediate_res/ins.csv",ins,fmt='%.5f', delimiter=',',header="t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,A1,A2,A3")
    np.savetxt("intermediate_res/q.csv", q, fmt='%.5f', delimiter=',',header="Q1,Q2,Q3,Q4,Q5")
    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    #dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'model/dqn_{}_weights_{}.h5f'.format(args.env_name,suffix)
    if args.weights:
        weights_filename = 'model/dqn_{}_weights_{}.h5f'.format(args.weights,suffix)
    dqn.load_weights(weights_filename)
    #dqn.test(env, nb_episodes=10, visualize=False)
    dqn.test(env, nb_max_episode_steps=args.testing_steps,visualize=False)
    '''
    import Gnuplot
    gp = Gnuplot.Gnuplot(persist=3)
    # gp('set terminal x11 size 350,225')
    # gp('set pointsize 2')
    # gp('set yrange [0.0:0.05]')
    plot1 = Gnuplot.PlotItems.Data(env.price_data, with_="linespoints lt rgb 'green' lw 2 pt 7", title="price")
    plot2 = Gnuplot.PlotItems.Data(env.action_data, with_="linespoints lt rgb 'blue' lw 2 pt 7", title="action")
    plot3 = Gnuplot.PlotItems.Data(env.treward_data, with_="linespoints lt rgb 'red' lw 2 pt 7", title="total_reward")
    gp.plot(plot3,plot2, plot1)
    epsFilename = args.env_name+'.eps'
    gp.hardcopy(epsFilename, terminal='postscript', enhanced=1, color=1)  # must come after plot() function
    gp.reset()
    x=np.random.uniform(0,1,22).reshape(22,1)
    print(dqn.forward(x))
    #
    # print env.price_data,len(env.price_data)
    # print env.treward_data,len(env.treward_data)
    # print env.action_data,len(env.action_data)
    '''
    import matplotlib.pyplot as plt

    price_data_x=[x for (x,y) in env.price_data]
    price_data_y=[y for (x,y) in env.price_data]

    treward_data_x=[x for (x,y) in env.treward_data]
    treward_data_y=[y for (x,y) in env.treward_data]

    action_data_x=[x for (x,y) in env.action_data]
    action_data_y=[y for (x,y) in env.action_data]

    plt.plot(price_data_x,price_data_y,label="price",color="green")
    plt.plot(treward_data_x, treward_data_y,label="total_reward",color="blue")
    plt.plot(action_data_x,action_data_y,label="action",color="red")
    plt.legend(loc='upper left')
    pic_name = "jpg/dqn_{}_{}.jpg".format(args.env_name,args.testing_path.split('/')[-1].split('.')[0])
    plt.savefig(pic_name)
    plt.show()
