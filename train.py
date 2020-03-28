

import tensorflow as tf
import numpy as np
import datetime
import data_provider
import common
import model as M
import dataset as D
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def get_train_dataset():
    data = data_provider.loadTrainData()
    return D.get_dataset(data)

def get_eval_dataset():
    data = data_provider.loadEvalData()
    return D.get_dataset(data)

# def get_debug_data():
#     return tf.data.Dataset.from_tensor_slices((np.zeros([32*381,512,3]), np.zeros([32*381,1])))

def train():

    if os.path.exists("deepannotate.h5"):
        model = tf.keras.models.load_model("deepannotate.h5")
    else:
        model = M.get_model_tf2(common.NUM_POINT, common.NUM_CLASSES[2], True)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=["sparse_categorical_accuracy"])
    model.summary()
    
    input_data = get_train_dataset()
    #input_data = input_data.shuffle(buffer_size=32*382)
    input_data = input_data.batch(32)

    eval_data = get_eval_dataset()
    eval_data = eval_data.batch(32)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def lr_scheduler(epoch):
        if epoch < 20:
            return 0.001
        elif epoch < 40:
            return 0.0005
        elif epoch < 80:
            return 0.0001
        elif epoch < 120:
            return 0.00005
        else: 
            return 0.00001

        #return max(0.001 * (0.7 ** (epoch / 10)), 0.00001)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

    class SaveCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                self.model.save("deepannotate.h5", include_optimizer=True, overwrite=True)
                print("model saved!")


    model.fit(input_data, validation_data=eval_data , epochs=250, callbacks=[tensorboard_callback, lr_callback, SaveCallback()])

    model.save("deepannotate.h5", include_optimizer=True, overwrite=True)


if __name__ == "__main__":
    train()