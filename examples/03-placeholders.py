# Import TensorFlow
import tensorflow as tf

# Define placeholder input tensors
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Define an "add" node (NumPy like syntax)
add_node = a + b  # + provides a shortcut for tf.add(a, b)

# Define a "multiply" node (NumPy like syntax)
add_and_double_node = add_node * 2.0  # * provides a shortcut for tf.mult(a, b)

# Initialize the session
with tf.Session() as sess:
    # Run the session with "rank 0" tensors as input
    print("\nRank 0 add:")
    print(sess.run(add_node, {a: 10, b: 2.5}))

    # Run the session with "rank 1" tensors as input
    print("\nRank 1 add:")
    print(sess.run(add_node, {a: [1, 5], b: [7, 3]}))

    # Run the session with "rank 0" tensors as input
    print("\nRank 0 add and double:")
    print(sess.run(add_and_double_node, {a: 10, b: 2.5}))

    # Run the session with "rank 1" tensors as input
    print("\nRank 1 add and double:")
    print(sess.run(add_and_double_node, {a: [1, 5], b: [7, 3]}))
