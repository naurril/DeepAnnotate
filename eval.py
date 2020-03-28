


import tensorflow as tf
import data_provider
import dataset as D

RESAMPLE_NUM = 5

model_path = "deepannotate.h5"
model = tf.keras.models.load_model(model_path)
model.summary()


def get_eval_dataset():
    data = data_provider.loadEvalData()
    return D.get_dataset(data)
eval_data = get_eval_dataset()
eval_data = eval_data.batch(32)
    
model.evaluate(eval_data)
