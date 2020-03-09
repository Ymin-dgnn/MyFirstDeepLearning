import tensorflow as tf

#X and y data (w=1, b=0)
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis Xw+b
hypothesis = x_train*w+b

# Cost function
cost = tf.reduce_mean( tf.square(hypothesis - y_train))

# Minimize cost (Gradient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

sess.run(tf.global_variables_initializer()) # for w and b 'Variables'

for step in range(2001):
    sess.run(train)
    if step %50 == 0: #50회의 iteration마다 출력
        print(step, sess.run(cost), sess.run(w), sess.run(b))
