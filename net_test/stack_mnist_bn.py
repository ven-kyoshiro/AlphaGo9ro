# coding:utf-8
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 10000, 'Number of epochs')
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_float('learning_rate', 1e-4, '')
flags.DEFINE_string('summary_dir', 'summary', '')
flags.DEFINE_string('output', 'checkpoint.ckpt', 'Output filename')

# this is a simpler version of Tensorflow's 'official' version. See:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
def batch_norm_wrapper(inputs, phase_train=None, decay=0.99):
  epsilon = 1e-5
  out_dim = inputs.get_shape()[-1]
  scale = tf.Variable(tf.ones([out_dim]))
  beta = tf.Variable(tf.zeros([out_dim]))
  pop_mean = tf.Variable(tf.zeros([out_dim]), trainable=False)
  pop_var = tf.Variable(tf.ones([out_dim]), trainable=False)

  if phase_train == None:
    return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

  rank = len(inputs.get_shape())
  axes = range(rank - 1)  # nn:[0], conv:[0,1,2]
  batch_mean, batch_var = tf.nn.moments(inputs, axes)

  ema = tf.train.ExponentialMovingAverage(decay=decay)

  def update():  # Update ema.
    ema_apply_op = ema.apply([batch_mean, batch_var])
    with tf.control_dependencies([ema_apply_op]):
      return tf.nn.batch_normalization(inputs, tf.identity(batch_mean), tf.identity(batch_var), beta, scale, epsilon)
  def average():  # Use avarage of ema.
    train_mean = pop_mean.assign(ema.average(batch_mean))
    train_var = pop_var.assign(ema.average(batch_var))
    with tf.control_dependencies([train_mean, train_var]):
      return tf.nn.batch_normalization(inputs, train_mean, train_var, beta, scale, epsilon)

  return tf.cond(phase_train, update, average)

def build_graph(is_training):
  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  x = tf.placeholder(tf.float32, shape=[None, 784])
  y_ = tf.placeholder(tf.float32, shape=[None, 10])
  phase_train = tf.placeholder(tf.bool, name='phase_train') if is_training else None

  x_image = tf.reshape(x, [-1, 28, 28, 1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  h_conv1 = conv2d(x_image, W_conv1)
  bn1 = batch_norm_wrapper(h_conv1, phase_train)
  h_pool1 = max_pool_2x2(tf.nn.relu(bn1))

  W_conv2 = weight_variable([5, 5, 32, 64])
  h_conv2 = conv2d(h_pool1, W_conv2)
  bn2 = batch_norm_wrapper(h_conv2, phase_train)
  h_pool2 = max_pool_2x2(tf.nn.relu(bn2))

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  bn_fc1 = batch_norm_wrapper(tf.matmul(h_pool2_flat, W_fc1), phase_train)
  h_fc1 = tf.nn.relu(bn_fc1)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])

  y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

  cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  return x, phase_train, y_, train_step, accuracy
  
def train(mnist):
  x, phase_train, y_, train_step, accuracy = build_graph(is_training=True)

  train_accuracy_summary = tf.scalar_summary('train accuracy', accuracy)
  test_accuracy_summary = tf.scalar_summary('test accuracy', accuracy)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)

    for step in range(FLAGS.epochs):
      batch = mnist.train.next_batch(FLAGS.batch_size)
      if step % 100 == 0:
        train_accuracy_result, train_accuracy = sess.run([train_accuracy_summary, accuracy], feed_dict={x: batch[0], phase_train: False, y_: batch[1]})
        test_accuracy_result, test_accuracy = sess.run([test_accuracy_summary, accuracy], feed_dict={x: mnist.test.images, phase_train: False, y_: mnist.test.labels})
        summary_writer.add_summary(train_accuracy_result, step)
        summary_writer.add_summary(test_accuracy_result, step)
        print("step %d, training accuracy %g, test accuracy %g" % (step, train_accuracy, test_accuracy))
      sess.run(train_step, feed_dict={x: batch[0], phase_train: True, y_: batch[1]})

    test_accuracy = sess.run(accuracy, feed_dict={
      x: mnist.test.images, phase_train: False, y_: mnist.test.labels})
    print("test accuracy %g" % test_accuracy)

    saver = tf.train.Saver()
    return saver.save(sess, FLAGS.output)


def test(mnist, saved_model):
  x, _, y_, train_step, accuracy = build_graph(is_training=False)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, saved_model)

    test_accuracy = sess.run(accuracy, feed_dict={
      x: mnist.test.images, y_: mnist.test.labels})
    print("test accuracy %g" % test_accuracy)


def main():
  mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
  saved_model = train(mnist)

  print('--------')
  tf.reset_default_graph()
  test(mnist, saved_model)


if __name__ == '__main__':
  main()
