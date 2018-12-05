import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    print("conv1=====",conv1.shape)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)
    print("conv1 pool=====",conv1.shape)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print("conv2=====",conv2.shape)
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)
    print("conv2====pool=",conv2.shape)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    print("conv3=====",conv3.shape)
    conv3 = maxpool2d(conv3, k=2)
    print("conv3=====",conv3.shape)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])

    print("fc1 flatten=====",fc1.shape)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print("fc1=====",fc1.shape)
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print("out=====",out.shape)
    return out


class Model:
    
    def __init__(self,lr):
        
        # TODO: Define network
        # ...
        self.lr = lr
        x = tf.placeholder(tf.float32, [None, 96, 96, 1],name="input")
        #change the Y placeholder dimension for brakes,acc etc
        y = tf.placeholder(tf.float32, [None, 5],name="output")
        weights = {
            'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W3', shape=(12*12*128,128), initializer=tf.contrib.layers.xavier_initializer()),
            #'wd1': tf.get_variable('W3', shape=(24 * 24 * 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(128, 5), initializer=tf.contrib.layers.xavier_initializer()),

            #'out': tf.get_variable('W6', shape=(64,5), initializer=tf.contrib.layers.xavier_initializer()),

        }
        biases = {
            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            #'bd1': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),

            'out': tf.get_variable('B4', shape=(5), initializer=tf.contrib.layers.xavier_initializer()),
        }

        pred = conv_net(x, weights, biases)
        save_pred = tf.identity(pred,"my_pred")


        # TODO: Loss and optimizer
        # ...
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        cost_name = tf.identity(cost,"my_cost")
        print("cost tensor ",cost)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost,name="my_adam")
        print("lr value=====>", self.lr)
        print("optimizer tensor",optimizer)
        #opt_name = tf.identity(optimizer,"my_optimizer")
        
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        cp_name = tf.identity(correct_prediction,"predict_output")

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_name = tf.identity(accuracy,"my_acc")
        
        init = tf.global_variables_initializer()
        
        # TODO: Start tensorflow session
        # ...
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver.save(self.sess, './SaveGraph/data-all')
        
        
    
        
        

        

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)

    def save(self, file_name):
        self.saver.save(self.sess, file_name)
