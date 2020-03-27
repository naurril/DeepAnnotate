

import tensorflow as tf
import numpy as np
import datetime
import data_provider
import common


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def transform_net(input, is_training):
    
    K = max(input.shape[2:])
    x = input

    x = tf.keras.layers.Conv2D(64, (1,x.shape[2]), strides=(1,1), padding='valid', activation=tf.nn.relu, data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), padding='valid', activation=tf.nn.relu, data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='valid', activation=tf.nn.relu, data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.MaxPool2D(pool_size=(x.shape[1], 1))(x)

    x = tf.squeeze(x, axis=[1,2])

    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Dense(K*K)(x)  # 3*3 matrix

    x = tf.add(x, tf.constant(np.eye(K).flatten(), dtype=tf.float32))
    x = tf.reshape(x, [-1,K,K])
    
    return x



def get_model_tf2(num_point, num_classes, is_training):
    input_pointcloud = tf.keras.Input(shape=(num_point, 3)) # batch_size is optional
    x = tf.expand_dims(input_pointcloud, -1)

    trans = transform_net(x, is_training)
    x = tf.matmul(input_pointcloud, trans)
    
    x = tf.expand_dims(x, -1)
    x = tf.keras.layers.Conv2D(64, (1,3), strides=(1,1), padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Conv2D(64, (1,1), strides=(1,1), padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)


    # transform feature here
    # B,512,1,64
    trans = transform_net(x, is_training)
    x = tf.matmul(tf.squeeze(x,[2]), trans)
    x = tf.expand_dims(x, [2])

    x = tf.keras.layers.Conv2D(64, (1,1), strides=(1,1), padding='valid', activation=tf.nn.relu, data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), padding='valid', activation=tf.nn.relu, data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='valid', activation=tf.nn.relu, data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)

    x = tf.keras.layers.MaxPool2D(pool_size=(x.shape[1], 1))(x)

    x = tf.squeeze(x, axis=[1,2])

    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)
    x = tf.keras.layers.Dropout(0.3)(x, is_training)

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    x = tf.keras.layers.BatchNormalization()(x, is_training)
    x = tf.keras.layers.Dropout(0.3)(x, is_training)

    x = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=input_pointcloud, outputs=x)
    return model



def get_data():
    data = data_provider.loadTrainData()
    #point_clouds = [x for x in map(lambda o: o["points"], data)]
    index_data = tf.data.Dataset.from_tensor_slices([x for x in range(len(data))])

    def sample(idx):
        sample = common.sample_one_input_data(data[idx], common.NUM_POINT)
        #print(sample["angle"])
        angle_class = common.angle_to_class(sample["angle"])
        return sample["points"], angle_class[2:] #only z angle, IMPORTANT: don't use scalar value

    def tf_rotate_object(obj_idx):
        [points, angle] = tf.py_function(sample, [obj_idx], [tf.float32, tf.float32])
        points.set_shape((common.NUM_POINT, 3))
        angle.set_shape((1))
        return points, angle

    
    input_data = index_data.map(tf_rotate_object)
    return input_data

def get_test_data():
    return tf.data.Dataset.from_tensor_slices((np.zeros([32*381,512,3]), np.zeros([32*381,1])))

def train():

    model = get_model_tf2(common.NUM_POINT, common.NUM_CLASSES[2], True)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["sparse_categorical_accuracy"])
    model.summary()
    
    input_data = get_data()
    input_data = input_data.shuffle(buffer_size=1000)
    input_data = input_data.batch(32)
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def lr_scheduler(epoch):
        if epoch < 5:
            return 0.001
        elif epoch < 10:
            return 0.0005
        elif epoch < 40:
            return 0.0001
        elif epoch < 80:
            return 0.00005
        else: 
            return 0.00001

        #return max(0.001 * (0.7 ** (epoch / 10)), 0.00001)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

    model.fit(input_data, epochs=250, callbacks=[tensorboard_callback, lr_callback])


def test_data():
    data = get_data()
    for p,a in data.take(1):
        print(p.numpy(), a.numpy())

if __name__ == "__main__":
    train()
    #test_data()
