from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *
import tensorflow as tf

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def back_to_id(a):
    if (a == LEFT): return [-1.0, 0.0, 0.0]                # LEFT: 1
    elif (a == RIGHT): return [1.0, 0.0, 0.0]             # RIGHT: 2
    elif (a == ACCELERATE): return [0.0, 1.0, 0.0]        # ACCELERATE: 3
    elif (a == BRAKE): return [0.0, 0.0, 0.2]              # BRAKE: 4
    else:       
        return [0.0, 0.0, 0.0]                                # STRAIGHT = 0

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()

    saver = tf.train.import_meta_graph('./models/agent.ckpt.meta')
    saver.restore(agent.sess, tf.train.latest_checkpoint('./models/'))
    graph = tf.get_default_graph()


    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        #    state = ...
        #print("state.shape BEFORE ...", state.shape)
        state = np.expand_dims(rgb2gray(state),axis=3)
        state = np.expand_dims(state,axis=0)
        #print("state.shape ...", state.shape)
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        # a = ...
        pred = agent.sess.graph.get_tensor_by_name('my_pred:0')
        #print("PRED...", pred)
        x = agent.sess.graph.get_tensor_by_name('input:0')
        a = agent.sess.run(pred, feed_dict={x: state})
        print(" command ", np.argmax(a[0]))
        #print(" command ", a[0])

        a = back_to_id(np.argmax(a[0]))

        if step<10:
            a = [0.0, 1.0, 0.0]
        # a = np.array([0.0, 1.0, 0.0])

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = Model(lr=0.1)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "./results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
