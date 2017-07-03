
import argparse
import sys
import tensorflow as tf
import os
from tensorflow.contrib.learn.python.learn.datasets import mnist

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(dataset,file):
    images = dataset.images
    labels = dataset.labels
    num = dataset.num_examples

    row=images.shape[1]
    col=images.shape[2]
    depth=images.shape[3]

    filename = os.path.join(FLAGS.record_dir,file+'.tfrecords')
    print("Writing into {}".format(filename))

    with tf.python_io.TFRecordWriter(filename) as tf_writer:
        for i in range(num):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'row':_int64_feature(row),
                'col':_int64_feature(col),
                'depth':_int64_feature(depth),
                'label':_int64_feature(int(labels[i])),
                'image_raw':_bytes_feature(image_raw)
            }))
            tf_writer.write(example.SerializeToString())
        tf_writer.close()

def main(_):
    # Get the data.
    data_sets = mnist.read_data_sets(FLAGS.data_dir,
                                     dtype=tf.uint8,
                                     reshape=False,
                                     validation_size=100)
    create_tfrecord(data_sets.train,'train')
    create_tfrecord(data_sets.validation,'validation')
    create_tfrecord(data_sets.test,'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--record_dir',type=str,default='/tmp/tensorflow/mnist/records',help='Directory for tfrecord')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
