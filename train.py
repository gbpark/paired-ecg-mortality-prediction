import glob
import tensorflow as tf
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
    parsed = tf.io.parse_single_example(element, parse_dic)
    return tf.io.parse_tensor(parsed['b_signal0'], tf.float32), \
           tf.io.parse_tensor(parsed['b_signal1'], tf.float32), \
           parsed['label0'], parsed['label1'], parsed['delta'], parsed['pid'], parsed['dt0'], parsed['dt1']

def prepare_sample(signal0, signal1, label0, label1, delta, pid, dt0, dt1):
    return (signal0, signal1), delta

def get_dataset(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE) \
        .map(_parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE) \
        .map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

if __name__ == '__main__':
    batch_size = 512
    MAX_EPOCHS = 200
    record_files = sorted(glob.glob('../datasets/paired-all-p*.tfrecords'))
    train_idx, val_idx = 6, 7
    train_files, val_files = record_files[:train_idx], record_files[train_idx:val_idx]
    train_dataset = get_dataset(train_files, batch_size)
    val_dataset = get_dataset(val_files, batch_size)
    model = get_compiled_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', save_best_only=True)
    ]
    model.fit(train_dataset, validation_data=val_dataset, epochs=MAX_EPOCHS, callbacks=callbacks)
