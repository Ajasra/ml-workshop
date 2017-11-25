# Import TensorFlow
import tensorflow as tf

# Define 3 "rank 0" constant tensors
# 3 - a rank 0 tensor; a scalar with shape []
# [1., 2., 3.] - a rank 1 tensor; a vector with shape [3]
# [[1., 2., 3.], [4., 5., 6.]] - a rank 2 tensor; a matrix with shape [2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]] - a rank 3 tensor with shape [2, 1, 3]
node1 = tf.constant(7.0, dtype=tf.float32)
node2 = tf.constant(8.0) # also tf.float32 implicitly
node3 = tf.constant(9.0)

# Print out the Tensors
print("Tensors:")
print(node1, node2, node3)
print("\n\n\n")

# Initialize the session
with tf.Session() as sess:
    # Store the session's result
    result = sess.run([node1, node2, node3])

    print("Session output:")
    print(result)
