
import argparse
import sys
import tensorflow as tf
import os

from tensorflow.examples.tutorials.mnist import mnist
import matplotlib.pyplot as plt



def read_and_decode(file_q):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(file_q)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'row': tf.FixedLenFeature([], tf.int64),
                                           'col': tf.FixedLenFeature([], tf.int64),
                                           'depth': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw':tf.FixedLenFeature([],tf.string)
                                       })
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    # Convert row from a scalar uint8 tensor to an int32 scalar.
    row = tf.cast(features['row'], tf.int32)
    # Convert col from a scalar uint8 tensor to an int32 scalar.
    col = tf.cast(features['col'], tf.int32)
    # Convert depth from a scalar uint8 tensor to an int32 scalar.
    depth = tf.cast(features['depth'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])


    return image,label,row,col,depth


def input(tf_file,epoch,batch_size):
    # returns a queue. adds a queue runner for the queue to the current graph's QUEUE_RUNNER
    filename_queue = tf.train.string_input_producer([tf_file], num_epochs=epoch)
    image,label,row,col,depth = read_and_decode(filename_queue)
    # return a list or dictionary. adds 1) a shuffling queue into which tensors are enqueued; 2) a dequeue_many operation to create batches
    # from the queue 3) a queue runner to QUEUE_RUNNER collection , to enqueue the tensors.
    data, labels,H,W,D = tf.train.shuffle_batch([image, label,row,col,depth], batch_size=batch_size, num_threads=2,
                                                      capacity=100 + 3 * batch_size, min_after_dequeue=1)
    # print(" in input image shape:{}".format(data.get_shape))
    # print(" in input lable shape:{}".format(labels.get_shape))
    return data, labels,H,W,D



def main(_):
    filename = FLAGS.record_dir+'/train.tfrecords'
    if not os.path.exists(filename):
        print("file:{} does not exists".format(filename))
        return 0
    batch =5
    data,label,row,col,depth = input(tf_file=filename,epoch=2,batch_size=batch)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        step = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                image,image_label,image_row,image_col,image_depth = sess.run(
                    [data,label,row,col,depth])
                print step
                print image.shape
                im_re=image.reshape([batch, image_row[0], image_col[0], image_depth[0]])
                print im_re.shape
                #print label.shape

                if (step > 30) and (step < 32):

                    for b in range(batch):
                        im = im_re[b]
                        la = image_label[b]
                        print("im shape:{}".format(im.shape))
                        print("im label:{}".format(la))
                        im_show=im.reshape([im.shape[0],im.shape[1]])
                        plt.imshow(im_show)


                step += 1
        except tf.errors.OutOfRangeError:
            print("Done. Epoch limit reached.")
        finally:
            coord.request_stop()
        coord.join(threads)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--record_dir',type=str,default='/tmp/tensorflow/mnist/records',help='Directory for tfrecord')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
