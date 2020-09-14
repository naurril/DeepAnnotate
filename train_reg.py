

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


model_file = "deepannotate_reg.h5"
weights_file = "da_weights_reg.h5"


def train():

    if os.path.exists(model_file):
        model = tf.keras.models.load_model("deepannotate_reg.h5")
    else:
        model = M.get_model_reg(common.NUM_POINT, 2, True)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), 
                    loss=tf.keras.losses.mse,
                    metrics=["mse"])
    model.summary()
    
    input_data = D.get_reg_train_dataset()
    #input_data = input_data.shuffle(buffer_size=32*382)
    input_data = input_data.batch(32)

    eval_data = D.get_reg_eval_dataset()
    eval_data = eval_data.batch(32)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def lr_scheduler(epoch):
        if epoch < 5:
            return 0.001
        elif epoch < 20:
            return 0.0005
        elif epoch < 40:
            return 0.0001
        elif epoch < 80:
            return 0.00005
        else: 
            return 0.00001
        #return max(0.001 * (0.7 ** (epoch / 10)), 0.00001)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

    class SaveCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                self.model.save(model_file, include_optimizer=True, overwrite=True)
                model.save_weights(weights_file)
                print("model saved!")


    model.fit(input_data, validation_data=eval_data , epochs=250, callbacks=[tensorboard_callback, lr_callback, SaveCallback()])

    model.save(model_file, include_optimizer=True, overwrite=True)
    model.save_weights(weights_file)


if __name__ == "__main__":
    train()