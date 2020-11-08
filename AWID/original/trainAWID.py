import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import os
import SNN

input = tf.placeholder(tf.float32)
input_exp = tf.exp(input)
groundtruth = tf.placeholder(tf.float32)

try:
    w1 = np.load('weight_awid1.npy')
    w2 = np.load('weight_awid2.npy')
    w3 = np.load('weight_awid3.npy')
    layer_in = SNN.SNNLayer(46, 100, w1)
    layer_out1 = SNN.SNNLayer(100, 100, w2)
    layer_out2 = SNN.SNNLayer(100, 4, w3)
    print('Weight loaded!')
except:
    layer_in = SNN.SNNLayer(46, 100)
    layer_out1 = SNN.SNNLayer(100, 100)
    layer_out2 = SNN.SNNLayer(100, 4)
    print('No weight file found, use random weight')

layerin_out = layer_in.forward(input_exp)
layerout_out1 = layer_out1.forward(layerin_out)
layerout_out2 = layer_out2.forward(layerout_out1)
nnout = tf.log(layerout_out2)

layerout_groundtruth = tf.concat([layerout_out2,groundtruth],1)
loss = tf.reduce_mean(tf.map_fn(SNN.loss_func,layerout_groundtruth))

wsc = layer_in.w_sum_cost() + layer_out1.w_sum_cost() + layer_out2.w_sum_cost()
l2c = layer_in.l2_cost() + layer_out1.l2_cost() + layer_out2.l2_cost()

K = 100
K2 = 1e-3
learning_rate = 1e-3
TRAINING_BATCH = 10

SAVE_PATH = os.getcwd() + '/weight_awid'

cost = loss + K*wsc + K2*l2c

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(cost)

config = tf.ConfigProto(device_count={'GPU': 1})
config.gpu_options.allow_growth = True
sess = tf.Session()
sess.run(tf.global_variables_initializer())

scale = 2
awid = SNN.Awid(path=["dataset/X_train.npy","dataset/y_train.npy"])

print('training started')
step = 1
while(True):
    xs, ys = awid.next_batch(TRAINING_BATCH, shuffle=True)
    xs = scale*xs
    [out,c,_] = sess.run([nnout,cost,train_op],{input:xs,groundtruth:ys})
    if step % 20 == 1:
        print('step '+repr(step) +', cost='+repr(c))
        w1 = sess.run(layer_in.weight)
        w2 = sess.run(layer_out1.weight)
        w3 = sess.run(layer_out2.weight)
        np.save(SAVE_PATH + '1', w1)
        np.save(SAVE_PATH + '2', w2)
        np.save(SAVE_PATH + '3', w3)
    step = step + 1

