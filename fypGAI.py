import tensorflow as tf
import numpy as np
import os
import pprint

def define_flags():
    """
    Define all the command-line parameters.

    Return:
      The FLAGS object.
    """
    # 定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数

    flags = tf.app.flags

    # 定义命令行参数，第一个是：参数名称，第二个是：参数默认值，第三个是：参数描述

    flags.DEFINE_integer("feature_size", 8, "Number of feature size")
    flags.DEFINE_integer("label_size", 2, "Number of label size")
    flags.DEFINE_string("file_format", "tfrecords", "Support tfrecords, csv")
    flags.DEFINE_string("train_files",
                        "/home/wr/fyp/result.csv.tfrecords",
                        "Train files which supports glob pattern")
    flags.DEFINE_string("validation_files",
                        "/home/wr/fyp/validation.csv.tfrecords",
                        "Validate files which supports glob pattern")
    flags.DEFINE_float("learning_rate", 0.0017, "Learning rate")
    flags.DEFINE_integer("epoch_number", 1500, "Number of epoches")
    flags.DEFINE_integer("train_batch_size", 50, "Batch size")
    flags.DEFINE_integer("validation_batch_size", 20, "validate Batch size")
    flags.DEFINE_integer("MAX_STEP", 1500, "Steps to train")
    flags.DEFINE_string("model_path", "/home/wr/fyp/saved model", "Path of the model")
    flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization")#Batch Normalization, 批标准化, 和普通的数据标准化类似, 是将分散的数据统一的一种做法, 也是优化神经网络的一种方法.
    flags.DEFINE_float("bn_epsilon", 0.001, "Epsilon of batch normalization")
    flags.DEFINE_string("output_path", "./tensorboard/", "Path for tensorboard")
    FLAGS = flags.FLAGS
    assert (FLAGS.file_format in ["tfrecords", "csv"])
    # Print flags
    parameter_value_map = {}
    for key in FLAGS.__flags.keys():
        parameter_value_map[key] = FLAGS.__flags[key].value
    pprint.PrettyPrinter().pprint(parameter_value_map)
    return FLAGS


FLAGS = define_flags()

def parse_tfrecords_function(example_proto):
  """
  Decode TFRecords for Dataset.

  Args:
    example_proto: TensorFlow ExampleProto object.

  Return:
    The op of features and labels
  """
  features = {
      "features": tf.FixedLenFeature([8], tf.float32),
      "label": tf.FixedLenFeature([], tf.int64, default_value=0)
  }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["features"], parsed_features["label"]


def parse_csv_function(line):
  """
  Decode CSV for Dataset.

  Args:
    line: One line data of the CSV.

  Return:
    The op of features and labels
  """

  FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], 
                    [0]]

  fields = tf.decode_csv(line, FIELD_DEFAULTS)

  label = fields[-1]
  label = tf.cast(label, tf.int64)
  features = tf.stack(fields[0:-1])

  return features, label

def full_connect(inputs,
                 weights_shape,
                 biases_shape,
                 is_train=True,
                 FLAGS=None):
  """
    Define full-connect layer with reused Variables.
    """

  weights = tf.get_variable(
      "weights", weights_shape, initializer=tf.random_normal_initializer())
  biases = tf.get_variable(
      "biases", biases_shape, initializer=tf.random_normal_initializer())
  layer = tf.matmul(inputs, weights) + biases #input 和 weights 矩阵相乘+bias

  if enable_bn and is_train:
    mean, var = tf.nn.moments(layer, axes=[0])
    scale = tf.get_variable(
        "scale", biases_shape, initializer=tf.random_normal_initializer())
    shift = tf.get_variable(
        "shift", biases_shape, initializer=tf.random_normal_initializer())
    layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                     bn_epsilon)
  return layer

def full_connect_relu(inputs,weights_shape,biases_shape,is_train=True,FLAGS=None):
      """
        Define full-connect layer and activation function with reused Variables. Implementing nonlinearization between input and output of neurons
        """

      layer = full_connect(inputs, weights_shape, biases_shape,is_train, FLAGS)
      layer = tf.nn.relu(layer)   #将每行的负值取0
      return layer
def inference(inputs,
                         input_units,
                         output_units,
                         is_train=True,
                         FLAGS=None):
  """
    Define the customed model.
    """

  hidden1_units = 128
  hidden2_units = 32
  hidden3_units = 8

  with tf.variable_scope("input_layer",reuse=tf.AUTO_REUSE):
    layer = full_connect_relu(inputs, [input_units, hidden1_units],
                              [hidden1_units], is_train, FLAGS)
  with tf.variable_scope("layer_0",reuse=tf.AUTO_REUSE):
    layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                              [hidden2_units], is_train, FLAGS)
  with tf.variable_scope("layer_1",reuse=tf.AUTO_REUSE):
    layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                              [hidden3_units], is_train, FLAGS)
  with tf.variable_scope("output_layer",reuse=tf.AUTO_REUSE):
    layer = full_connect(layer, [hidden3_units, output_units], [output_units],
                         is_train, FLAGS)
  return layer


def compute_softmax_and_accuracy(logits, labels):
      """
      Compute the softmax and accuracy of the logits and labels.

      Args:
        logits: The logits from the model.
        labels: The labels.

      Return:
        The softmax op and accuracy op.
      """
      softmax_op = tf.nn.softmax(logits)
      correct_prediction_op = tf.equal(tf.argmax(softmax_op, 1), labels)
      accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))

      return softmax_op, accuracy_op

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#feature_size=9
#label_size=2
#file_format= "tfrecords"
#train_files="./Users/WayneJ/PycharmProjects/application/data/cancer/cancer_test.csv",
#inference_result_file="./inference_result.txt"
#learning_rate=0.01
#epoch_number=1000
#train_batch_size=497
#MAX_STEP=15000
#model_path= "Users/WayneJ/PycharmProjects/final project/saved model"
#epoch_number = 1000
enable_bn=False  #Enable batch normalization
bn_epsilon=0.001, #Epsilon of batch normalization


train_buffer_size = FLAGS.train_batch_size * 3  #buffer_size: 一个tf.int64标量tf.Tensor，代表将被加入缓冲器的元素的最大数
validate_buffer_size = FLAGS.validation_batch_size * 3
train_filename_list = [filename for filename in FLAGS.train_files.split(",")]
train_filename_placeholder = tf.placeholder(tf.string, shape=[None])
if FLAGS.file_format == "tfrecords":
    train_dataset = tf.data.TFRecordDataset(train_filename_placeholder)
    train_dataset = train_dataset.map(parse_tfrecords_function).repeat(
        FLAGS.epoch_number).batch(FLAGS.train_batch_size).shuffle(
            buffer_size=train_buffer_size)
elif FLAGS.file_format == "csv":
    # Skip the header or not
    train_dataset = tf.data.TextLineDataset(train_filename_placeholder)
    train_dataset = train_dataset.map(parse_csv_function).repeat(
        FLAGS.epoch_number).batch(FLAGS.train_batch_size).shuffle(
            buffer_size=train_buffer_size)
train_dataset_iterator = train_dataset.make_initializable_iterator()
train_features_op, train_label_op = train_dataset_iterator.get_next()
validation_filename_list = [filename for filename in FLAGS.validation_files.split(",")]

validation_filename_placeholder = tf.placeholder(tf.string, shape=[None])
if FLAGS.file_format == "tfrecords":
    validation_dataset = tf.data.TFRecordDataset(
        validation_filename_placeholder)
    validation_dataset = validation_dataset.map(
        parse_tfrecords_function).repeat(FLAGS.epoch_number).batch(
        FLAGS.validation_batch_size).shuffle(
        buffer_size=validate_buffer_size)
elif FLAGS.file_format == "csv":
    validation_dataset = tf.data.TextLineDataset(
        validation_filename_placeholder)
    validation_dataset = validation_dataset.map(parse_csv_function).repeat(
        FLAGS.epoch_number).batch(FLAGS.validation_batch_size).shuffle(
        buffer_size=validate_buffer_size)
validation_dataset_iterator = validation_dataset.make_initializable_iterator(
)
validation_features_op, validation_label_op = validation_dataset_iterator.get_next(
)

input_units =FLAGS.feature_size
output_units = FLAGS.label_size
logits = inference(train_features_op, input_units, output_units, True)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=train_label_op)
loss = tf.reduce_mean(cross_entropy, name="loss")
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = FLAGS.learning_rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # 优化器
train_op = optimizer.minimize(loss, global_step=global_step)
# Define accuracy op op
train_accuracy_logits = inference(train_features_op, input_units,
                                  output_units, False)
train_softmax_op, train_accuracy_op = compute_softmax_and_accuracy(
    train_accuracy_logits, train_label_op)
validate_accuracy_logits = inference(validation_features_op, input_units,
                                  output_units, False)
validate_softmax_op, validate_accuracy_op = compute_softmax_and_accuracy(
    validate_accuracy_logits, validation_label_op)
# Initialize saver and summary and create session to run
sess = tf.Session()
tf.summary.scalar("loss", loss)
tf.summary.scalar("train_accuracy", train_accuracy_op)
tf.summary.scalar("validate_accuracy", validate_accuracy_op)
writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
init_op = [
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
]
sess.run(init_op)
sess.run([train_dataset_iterator.initializer],feed_dict={train_filename_placeholder: train_filename_list,})
sess.run([validation_dataset_iterator.initializer],feed_dict={validation_filename_placeholder: validation_filename_list,})

try:
    for step in np.arange(FLAGS.MAX_STEP):

        _, tra_loss, tra_acc, validate_acc = sess.run([train_op, loss, train_accuracy_op, validate_accuracy_op])
        if step % 100 == 0:
            print("Step %d, train loss = %.2f, train accuracy = %.2f, validate accuracy = %.2f" % (step, tra_loss, tra_acc, validate_acc))
            summary_str = sess.run(summary_op)
            writer.add_summary(summary_str, step)
            if step % 2000 == 0 or (step + 1) == FLAGS.MAX_STEP:
                checkpoint_path = os.path.join(FLAGS.model_path, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached.")

sess.close()
