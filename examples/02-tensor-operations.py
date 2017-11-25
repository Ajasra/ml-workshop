# Import TensorFlow
import tensorflow as tf

# Define 2 "rank 0" constant tensors
node1 = tf.constant(7.0, dtype=tf.float32)
node2 = tf.constant(8.0) # also tf.float32 implicitly

# Define an "add" node operation
node3 = tf.add(node1, node2)

# Print out the Tensors (not the actual values)
print("Operation Tensor:")
print(node3)
print("\n\n\n")

# Initialize the session
with tf.Session() as sess:
    # Store the session's result
    result = sess.run(node3)

    print("Session output:")
    print(result)
