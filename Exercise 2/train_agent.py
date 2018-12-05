from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
import tensorflow as tf
from datetime import datetime
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid

def minibatched(data, batch_size):
    assert len(data) % batch_size == 0, ("Data length {} is not multiple of batch size {}"
                                         .format(len(data), batch_size))
    return data.reshape(-1, batch_size, *data.shape[1:])


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train = np.expand_dims(rgb2gray(X_train),axis=3)
    #print("YTR ",y_train)
    y_train = action_to_id_all(y_train)
    y_train = one_hot_encoding(y_train,5)
    #print("YTRRRRRRRRRRRRRRRRRRRR",y_train.shape)
    #print("YTR__ ",y_train[12000])

    fname = "./results_bc_agent-%s.txt" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    for i in range(y_train.shape[0]):
        fh.write(str(i) + " -- " + str(y_train[i]) + "\n")
    fh.close()

    X_valid = np.expand_dims(rgb2gray(X_valid),axis=3)
    y_valid = action_to_id_all(y_valid)
    y_valid = one_hot_encoding(y_valid,5)
    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train, X_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(lr)
    tensorboard_eval = Evaluation(tensorboard_dir)
    #train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)
    
    batch_size = X_train.shape[0] // n_minibatches
    
    print("Batchsize",batch_size)
    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    saver = tf.train.import_meta_graph('SaveGraph/data-all.meta')
    saver.restore(agent.sess, tf.train.latest_checkpoint('./SaveGraph'))
    graph = tf.get_default_graph()

    x = agent.sess.graph.get_tensor_by_name('input:0')
    
    y = agent.sess.graph.get_tensor_by_name('output:0')
    #op = agent.sess.graph.get_operations()
    #print("before-----------------opti---opertaions",op)
    
    cost = agent.sess.graph.get_tensor_by_name('my_cost:0')
    optimizer = agent.sess.graph.get_operation_by_name('my_adam')

    
    accuracy = agent.sess.graph.get_tensor_by_name('my_acc:0')
    correct_prediction = agent.sess.graph.get_tensor_by_name('predict_output:0')

    summary_writer = tf.summary.FileWriter('./Output', agent.sess.graph)
    count = 0
    loss = 0
    acc_tr = 0
    valid_acc = 0
    for batch in range(X_train.shape[0] // batch_size):
        batch_x = X_train[batch*batch_size:min((batch+1)*batch_size,len(X_train))]
        batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))] 
        
        print("BATCH  ",batch)
        #print("Y BATCH SHAPE",batch_y.shape)
        # Run optimization op (backprop).
        # Calculate batch loss and accuracy
        opt = agent.sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        loss, acc_tr = agent.sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        if batch%20 == 0:
            te_dict = {"loss": loss, "training_accuracy": acc_tr}
            tensorboard_eval.write_episode_data(batch, te_dict)
            
    for batch in range(X_valid.shape[0] // batch_size):
        batch_x = X_valid[batch*batch_size:min((batch+1)*batch_size,len(X_valid))]
        batch_y = y_valid[batch*batch_size:min((batch+1)*batch_size,len(y_valid))]    
        # Run optimization op (backprop).
        # Calculate batch loss and accuracy
        cp = agent.sess.run(correct_prediction, feed_dict={x: batch_x, y: batch_y})
            
        count += len(cp[cp==True])
        valid_acc = count / len(X_valid)


        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #loss, acc  = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y})



            

    #print(" Loss= " +\"{:.6f}".format(loss) + ", Training Accuracy= " + \"{:.5f}".format(acc))
    print(loss, acc_tr)
    print("Validation Accuracy :",valid_acc)

    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #tensorboard_eval.write_episode_data(count, te_dict)
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print(model_dir)
    print("Model saved in file: %s" % model_dir)
    agent.sess.close()

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")
    print("X shape", X_valid.shape)
    print("y shape", y_valid.shape)
    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
    print("X shape", X_valid.shape)
    print("y shape", y_valid.shape)
    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, n_minibatches=1000, batch_size=32, lr=0.001, model_dir="./models/")
