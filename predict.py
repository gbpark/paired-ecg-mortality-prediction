import numpy as np
import tensorflow as tf
from sklearn import metrics
from model import get_compiled_model

def _parse_tfr_element(element):
    # 동일한 _parse_tfr_element 함수 재사용
    pass

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
    test_files = sorted(glob.glob('../datasets/paired-all-p*.tfrecords'))[7:]  # Test 데이터 파일
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
