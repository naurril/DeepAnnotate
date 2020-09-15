

import tensorflow as tf
import numpy as np
import datetime
import data_provider
import common
import model as M
import dataset as D
import os
import util

util.config_gpu()
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# def get_debug_data():
#     return tf.data.Dataset.from_tensor_slices((np.zeros([32*381,512,3]), np.zeros([32*381,1])))


def train():

    if os.path.exists("deepannotate_rp_discrimination.h5"):
        model = tf.keras.models.load_model("deepannotate_rp_discrimination.h5")
    else:
        model = M.get_model_rp_discrimination(common.NUM_POINT, 2, True)   # we have only 2 classes, positive/negative
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.keras.metrics.sparse_categorical_accuracy, tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)])
    model.summary()
    
    input_data = D.get_rp_train_dataset()
    #input_data = input_data.shuffle(buffer_size=32*382)
    input_data = input_data.batch(32)

    eval_data = D.get_rp_val_dataset()
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
                self.model.save("deepannotate_rp_discrimination", include_optimizer=True, overwrite=True)
                model.save_weights("da_rp_weights.h5")
                print("model saved!")


    model.fit(input_data, validation_data=eval_data , epochs=100, callbacks=[tensorboard_callback, lr_callback, SaveCallback()])

    model.save("deepannotate_rp_discrimination.h5", include_optimizer=True, overwrite=True)
    model.save_weights("da_rp_weights.h5")


if __name__ == "__main__":
    train()