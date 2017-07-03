import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def bytes_feature(values):
    """Encodes an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))



def encode_example(im,label):
    feature = {
        'label':bytes_feature(label.tostring()),
        'image':bytes_feature(im.tostring())
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    return example.SerializeToString()

def create_tf_record(image_file,label_file,tf_file):
    print(os.path.abspath(tf_file))

    with tf.python_io.TFRecordWriter(tf_file) as tf_writer:
        for i, (depth, label) in tqdm(enumerate(zip(image_file,label_file)),total=len(image_file)):
            if i is 0:
                print("depth shape:{}".format(depth.shape))
                print("label shape:{}".format(label.shape))

            example = encode_example(depth,label)
            tf_writer.write(example)

def read_and_decode(filename_queue,target_size,label_shape):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = {
        'label':tf.FixedLenFeature([],tf.string),
        'image':tf.FixedLenFeature([],tf.string),
    })
    #convert from a scalor string tensor
    label = tf.decode_raw(features['label'],tf.float32)

    image = tf.decode_raw(features['image'],tf.float32)

    #Need to reconstruct channels first then transpose channels
    image = tf.reshape(image,np.asarray(target_size))
    label_f = tf.reshape(label,np.asarray(label_shape))
    #label.set_shape(label_shape)

    #print(" after reshape image shape:{}".format(image.get_shape().as_list()))
    #print(" after reshape lable shape:{}".format(label.get_shape().as_list()))

    return label_f, image

def dataloader(tfrecord_file,num_epochs,image_target_size,label_shape,batch_size):
    with tf.name_scope('input'):
        if os.path.exists(tfrecord_file) is False:
            print("{} not exists".format(tfrecord_file))
        # returns a queue. adds a queue runner for the queue to the current graph's QUEUE_RUNNER
        filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=num_epochs)
        label,image = read_and_decode(filename_queue=filename_queue,target_size=image_target_size,label_shape = label_shape)
        print("label size:{}, image_size:{}".format(label.get_shape(),image.get_shape()))
        # return a list or dictionary. adds 1) a shuffling queue into which tensors are enqueued; 2) a dequeue_many operation to create batches
        # from the queue 3) a queue runner to QUEUE_RUNNER collection , to enqueue the tensors.
        data, labels= tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                          num_threads=1, capacity=10000 + 3 * batch_size,
                                                          min_after_dequeue=1)
        print("data size:{}, labels size:{}".format(data.get_shape(),labels.get_shape()))
    return data,labels
