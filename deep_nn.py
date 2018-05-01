
import tensorflow as tf
import numpy as np

x_test = [[1,3,2],[3,2,1],[0,2,1],[1,2,2],[4,2,3],[2,2,3]]
y_test = [[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,0,1],[0,1,0]]

X = tf.placeholder(tf.float32, [None,3])
y = tf.placeholder(tf.float32, [None,3])  #already in hot-encoded form
W = tf.Variable(tf.random_normal([3,3])) #in-dim, out-dim
b = tf.Variable(tf.random_normal([3])) #out-dim

hypo = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.arg_max(hypo, 1)
is_correct = tf.equal(prediction, tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(2000):
        cost_val, W_val, _ = sess.run([cost, W, optimizer], 
                                      feed_dict = {X:x_test, y:y_test})
        if step%500==0:
            print(step, cost_val, W_val)
        
    print("Prediction:", sess.run(prediction, feed_dict = {X:x_test}))
    print("Accuracy:", sess.run(accuracy, feed_dict = {X:x_test, y:y_test}))


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#simple NN
nb_classes=10
X= tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, nb_classes])
W = tf.Variable(tf.random_normal([784,nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypo = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

prediction = tf.arg_max(hypo, 1)
#is_correct = tf.equal(prediction, tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1), prediction), tf.float32))

training_epochs = 15
batch_size=100

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict = {X:batch_xs, y:batch_ys})
            avg_cost+=c/total_batch
            
        print(accuracy.eval(session=sess, 
               feed_dict={X:mnist.test.images, y:mnist.test.labels}))


#2개의 레이어를 가진 NN (hidden layer 노드 수 10개)
W1= tf.get_variable("W1", shape=[784,256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.get_variable("W2", shape=[256,20], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([20]))
layer2 = tf.sigmoid(tf.matmul(layer1, W2)+b2)

W3 = tf.get_variable("W3", shape=[20,nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypo = tf.nn.softmax(tf.matmul(layer2, W3)+b3)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypo),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

prediction = tf.arg_max(hypo,1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y,1),prediction),dtype=tf.float32))

training_epochs = 15  #몇 번 전체 데이터를 써서 트레이닝 시킬 것인가
batch_size=100

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)  #한 번 에포크를 돌리기 위해 반복해야 할 수(전체데이터수/한번트레이닝시킬수)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  #batch_size 만큼의 데이터 포인트만 가져온다
            c, _ = sess.run([cost, optimizer], feed_dict = {X:batch_xs, y:batch_ys})
            avg_cost+=c/total_batch
            
        print(accuracy.eval(session=sess, 
               feed_dict={X:mnist.test.images, y:mnist.test.labels}))


#learning rate...(?) overshooting vs local minima
#NaN의 저주..또 다른 이유: not normalized dataset
