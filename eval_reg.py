


import tensorflow as tf
import data_provider
import dataset as D
import model as M
import common
import util

util.config_gpu()

RESAMPLE_NUM = 5

weights_path = "../DeepAnnotate/models/da_weights_reg.h5"
model = M.get_model_reg(common.NUM_POINT, 2, False)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                    loss=tf.keras.losses.mse,
                    metrics=[tf.keras.metrics.mse])
model.load_weights(weights_path)

model.summary()

eval_data = D.get_reg_eval_dataset()
eval_data = eval_data.batch(32)
    
model.evaluate(eval_data)

