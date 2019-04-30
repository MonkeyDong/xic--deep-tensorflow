import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

#print(sys.argv)
X_tr = np.load("save_npy/x_train.npy") #(n,向量长度，通道数) (n,128,6)
X_vld = np.load("save_npy/x_ver.npy") 
y_tr = np.load("save_npy/y_train.npy") #one_hot编码向量 (n,)
y_vld =  np.load("save_npy/y_ver.npy")

X_test = np.load("save_npy/X_test.npy")
y_test = np.load("save_npy/y_test.npy")


batch_size = 64       # Batch size
seq_len = 256          # Number of steps
learning_rate = 0.00001
epochs = 1000

n_classes = 2
n_channels = 32


graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.int32, [None,1], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')


with graph.as_default():
    # (batch, 256, 16) --> (batch, 128, 64)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
    # (batch, 128, 32) --> (batch, 64, 128)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=128, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
    # (batch, 64, 64) --> (batch, 32, 256)
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=256, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
    
    # (batch, 32, 128) --> (batch, 16, 512)
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=512, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

    # (batch, 16, 256) --> (batch, 8, 1024)
    conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=1024, kernel_size=2, strides=1, 
                             padding='same', activation = tf.nn.relu)
    max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')


with graph.as_default():
    # Flatten and add dropout
    flat = tf.reshape(max_pool_5, (-1, 8*1024))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    
    # Predictions
    flat2 = tf.layers.dense(flat, 1024)
    flat2 = tf.nn.dropout(flat2, keep_prob=keep_prob_)

    logits = tf.layers.dense(flat2, n_classes)    
    
    logits = tf.sigmoid(logits)
    predict = tf.arg_max(logits,1,name='predict')
    cost = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=labels_)
    cost = tf.reduce_mean(cost)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    acc_, accuracy = tf.metrics.accuracy(labels=labels_,predictions=predict)    



validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

def get_batches(X, y, batch_size):
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(epochs):
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            
            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            if (iteration%10 == 0):                
                
                # Feed
                feed = {inputs_ : X_test, labels_ : y_test, keep_prob_ : 1.0}    
                # acc
                acc_ts = sess.run([accuracy], feed_dict = feed)                    
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Test acc: {:.6f}".format(np.mean(acc_ts)))
               
            # Iterate 
            iteration += 1
    
    saver.save(sess,"checkpoints-cnn/ms.ckpt")

'''
# Plot training and test loss
t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
'''
with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn'))

    feeds = {inputs_: X_test,
             labels_: y_test,
             keep_prob_: 1}		
    logit = sess.run(logits,feed_dict = feeds)
    t = sys.argv[1]
    np.save("output_npy/output"+t+".npy",logit)
    np.save("output_npy/labels"+t+".npy",y_test)
