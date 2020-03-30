


import tensorflow as tf
import data_provider
import dataset as D
import model as M
import common
import util

util.config_gpu()

RESAMPLE_NUM = 5

weights_path = "../DeepAnnotate/models/da_weights.h5"
model = M.get_model_cls(common.NUM_POINT, common.NUM_CLASSES[2], False)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.sparse_categorical_accuracy, tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)])
model.load_weights(weights_path)

model.summary()

eval_data = D.get_cls_eval_dataset()
eval_data = eval_data.batch(32)
    
model.evaluate(eval_data)

