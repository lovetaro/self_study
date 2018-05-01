
import tensorflow as tf
import numpy as np

#Basic Tensor calling
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3= tf.add(node1,node2)
session = tf.Session()
added = session.run(node3)
print(added) #7.0

#Placeholders

X_data = [[40,50,20],[60,30,50], [90,80,50]]
y_data = [[150],[130],[120]]

X = tf.placeholder(tf.float32, shape=[None, 3])  # 3*num of dataset(임의의 개수==None)
y = tf.placeholder(tf.float32, shape=[None,1])   # 1*num of dataset
W= tf.Variable(tf.random_normal([3,1]), name= 'weight')  #1*3 형태 Matrix
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = X*W + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(2000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                  feed_dict = {X:X_data, y:y_data})
    if step%10==0:
        print(step, "cost:", cost_val, "\nprediction:", hy_val)


#Taking care of slicing and np.array
xy = np.loadtxt('data-01-test-score.csv', delimiter= ',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None,3]) 
y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x*W+b


#Logistic classifier
#1. Training Data
x_data = [[1,2],[3,4],[3,1],[4,3],[2,3]]
y_data = [[0],[0],[1],[1],[0]] #0 or 1

#2. Placeholders for a tensor that will be fed(always)
X = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([2,1]), name='weight') #W*X 를 해서 y 를 도출하기 때문에 그 shape 에 주의해야. 그 모양은 [X의 차원,결과값(활성값)의 차원] 이 된다
b = tf.Variable(tf.random_normal([1]), name='bias')

hypo = tf.sigmoid(tf.matmul(X,W) + b)
cost = -tf.reduce_mean(y*tf.log(hypo)+(1-y)*tf.log(1-hypo))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypo>0.5, dtype=tf.float32)  #casting -  (T/F)표현을 0 또는  1 로 해줌 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(10000):
    cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, y:y_data})
    if step%200==0:
        print(step, cost_val)
h, c, a = sess.run([hypo, predicted,accuracy],
                  feed_dict = {X:x_data, y:y_data})
print(h, c, a)


#Softmax Classifier
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

"""
Softmax Classifier - binary 아닌 n 개의 카테고리 예측시에 사용! 
cross-entropy cost function(logistic cost)
sum(L(i)*Y(i), element-wise calculation 내적)
"""
x_sm = [[1,2,1,1,],[2,1,3,2],[4,2,4,2],[4,5,2,3],[2,3,1,1]]
y_sm = [[0,0,1],[1,0,0],[0,1,0],[0,1,0],[1,0,0]]  #one-hot-encoding

X_2 = tf.placeholder(tf.float32, [None,4])
y_2 = tf.placeholder(tf.float32,[None,3])
nb_classes_2 = 3           #according to Y vector dimension

W_2 = tf.Variable(tf.random_normal([4,nb_classes_2]),name='weight')
b_2 = tf.Variable(tf.random_normal([nb_classes_2]), name='bias')

sm_hypo = tf.nn.softmax(tf.matmul(X_2,W_2)+b_2) #라이브러리에서 바로 제공(하드코딩 불필요)
sm_cost = tf.reduce_mean(-tf.reduce_sum(y_2*tf.log(sm_hypo), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(sm_cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in range(2000):
    sess.run(optimizer, feed_dict={X_2:x_sm, y_2:y_sm})
    if step%200==0:
        print(step, sess.run(cost, feed_dict={X_2:x_sm, y_2:y_sm}))
        
a = sess.run(sm_hypo, feed_dict={X_2:[1,11,7,9]})
print(a, sess.run(tf.arg_max(a,1)))

all = sess.run(sm_hypo, feed_dict={X_2:[[1,11,6,5],
                                      [1,3,4,3],
                                      [1,1,0,1]]})   # (3,4)  4차원 벡터 3개 있는 데이터셋
                                      
print(all, sess.run(tf.arg_max(all,1)))  #axis=1. 각 열에서 최대값.. 4개의 y prediction  도출
