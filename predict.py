import numpy as np
import tensorflow as tf
from sklearn import metrics
from model import get_compiled_model

def _parse_tfr_element(element):
    parse_dic = {
        'b_signal0': tf.io.FixedLenFeature([], tf.string),
        'b_signal1': tf.io.FixedLenFeature([], tf.string),
        'label0': tf.io.FixedLenFeature([], tf.float32),
        'label1': tf.io.FixedLenFeature([], tf.float32),
        'pid': tf.io.FixedLenFeature([], tf.string),
        'dt0': tf.io.FixedLenFeature([], tf.string),
        'dt1': tf.io.FixedLenFeature([], tf.string),
        'delta': tf.io.FixedLenFeature([], tf.float32),
    }
    example_message = tf.io.parse_single_example(element, parse_dic)

    b_signal0 = example_message['b_signal0']
    b_signal1 = example_message['b_signal1']
    signal0 = tf.io.parse_tensor(b_signal0, out_type=tf.float32)
    signal1 = tf.io.parse_tensor(b_signal1, out_type=tf.float32)

    pid = example_message['pid']
    dt0 = example_message['dt0']
    dt1 = example_message['dt1']
    delta = example_message['delta']
        
    label0 = example_message['label0']
    label1 = example_message['label1']
    
    return signal0, signal1, label0, label1, delta, pid, dt0, dt1

def prepare_sample(signal0, signal1, label0, label1, delta, pid, dt0, dt1):
    return (signal0, signal1), delta
    
def prepare_rev_sample(signal0, signal1, label0, label1, delta, pid, dt0, dt1):
    return (signal1, signal0), -delta
    
def get_dataset(filenames, batch_size):
    pos_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)\
                .map(_parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)\
                .map(prepare_sample)
    rev_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)\
                .map(_parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)\
                .map(prepare_rev_sample)

    dataset = tf.data.Dataset.concatenate(pos_dataset, rev_dataset)
    dataset = dataset.shuffle(batch_size * 10, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def prepare_test_sample(signal0, signal1, label0, label1, delta, pid, dt0, dt1):
    return (signal0, signal1), delta

def get_test_dataset(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE) \
        .map(_parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(prepare_test_sample) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    batch_size = 512
    test_files = sorted(glob.glob('../datasets/paired-all-p*.tfrecords'))[7:]
    test_dataset = get_test_dataset(test_files, batch_size)
    model = get_compiled_model()
    model.load_weights('model_checkpoint.h5')
    preds, labels = [], []
    for data in test_dataset:
        pred = model.predict(data[0])
        preds.extend(pred)
        labels.extend(data[1].numpy())
    print('MSE:', metrics.mean_squared_error(labels, preds))
    print('MAE:', metrics.mean_absolute_error(labels, preds))
