from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

with tf.name_scope('inputData'):
    x = tf.placeholder(tf.float32, shape=[None, 784])

with tf.name_scope('inputLabels'):
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight Initialization 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# First layer: 
# The convolutional will compute 32 features for each 5x5 patch.
# [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the
# number of input channels, and the last is the number of output channels.
# We will also have a bias vector with a component for each output channel.
with tf.name_scope('FirstLayer'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # Reshape x to a 4d tensor, value, width, height and number of color channels.
    x_image = tf.reshape(x, [-1,28,28,1])
    # feed images to tensorboard (20 images)
    #tf.image_summary('input', x_image, 20)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# Second Layer
with tf.name_scope('SecondLayer'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Image size from 28x28 to now 7x7

# Now that the image size has been reduced to 7x7, we add a fully-connected
# layer with 1024 neurons to allow processing on the entire image
# We reshape the tensor from the pooling layer into a batch of vectors,
# multiply by a weight matrix, add a bias, and apply a ReLU.
with tf.name_scope('FullyConnectedLayer'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])  
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
with tf.name_scope('GetOutputsSoftmax'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # Softmax layer
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Train and Evaluate the Model
with tf.name_scope('GetEntropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    
with tf.name_scope('TrainStep'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('GetAccuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tensorboard
tf.scalar_summary('Cross entropy', cross_entropy)
tf.scalar_summary('Training accuracy', accuracy)
#tf.scalar_summary('sparsity', tf.nn.zero_fraction(h_fc1))
merged_summary_op = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('/tmp/mnist_logs/train', sess.graph)
test_writer = tf.train.SummaryWriter('/tmp/mnist_logs/test')

# initialize
sess.run(tf.initialize_all_variables())

# Training
for i in range(10000):
  batch = mnist.train.next_batch(50)
  if i%25 == 0:
    summary, acc = sess.run([merged_summary_op, accuracy], 
                            feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    test_writer.add_summary(summary, i)
    print("step %d, training accuracy %g"%(i, acc))
  else:
    summary, _ = sess.run([merged_summary_op, train_step],
                          feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})
    train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# save predictions to file
#test = sess.run(h_pool2, feed_dict={x: mnist.test.images, keep_prob: 1.0})
prediction = tf.argmax(y_conv,1).eval(feed_dict={x: mnist.test.images, keep_prob: 1.0})
real_labels = tf.argmax(mnist.test.labels, 1).eval()
errors = prediction[prediction != real_labels]
correct = real_labels[prediction != real_labels]
errors_images = mnist.test.images[prediction != real_labels]
errors_images = sess.run(tf.reshape(errors_images, [-1,28,28,1]))
image_summary = tf.image_summary('error', errors_images, 10).eval()
error_writer = tf.train.SummaryWriter('/tmp/mnist_logs/errors')
error_writer.add_summary(image_summary, 1)
# tensorboard --/tmp/mnist_logs

# Saving session
#trainDir = "samples/tensorFlow/train/"
#saver = tf.train.Saver()
#saver.save(sess, trainDir, global_step=i)

# Restoring session
#saver.restore(sess, "samples/tensorFlow/train/")
#saver.restore(sess, "-9999")
import pandas as pd
df = pd.DataFrame({'predictedLabels': prediction, 'realLabels': real_labels})
df.to_csv('samples/tensorFlow/results.csv')
#f=open('prediction.txt','w')  
#s1='\n'.join(str(x) for x in prediction)  
#f.write(s1)  
#f.close()  
