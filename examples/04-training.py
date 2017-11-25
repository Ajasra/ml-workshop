# Import TensorFlow
import tensorflow as tf

# Define variable tensors
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Define a placeholder input and output tensors
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Define a linear regression model
linear_model = W*x + b

# Calculate a loss function, the sum of squares differences between modeled x and y
squared_deltas = tf.square(linear_model - y)
loss_fn = tf.reduce_sum(squared_deltas)

# Define an optimizer function (using the default Gradient Descent)
optimizer_fn = tf.train.GradientDescentOptimizer(0.01)

# Define a training function that will update the variables
train_fn = optimizer_fn.minimize(loss_fn)

# Initialize the session
with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Run the training cycle 1000 times
    for i in range(1000):
        sess.run(train_fn, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

    # Run the session to get the trained variable values
    result = sess.run([W, b, loss_fn], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print(result)
