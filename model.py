
import tensorflow as tf


def conv2d(x, filters, kernel_size, strides, is_training):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='valid', data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x, is_training)
    x = tf.nn.relu(x)
    return x

def fc(x, units, is_training):
    x = tf.keras.layers.Dense(units)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x, is_training)
    x = tf.nn.relu(x)
    return x

def transform_net(input, is_training):
    
    K = max(input.shape[2:])
    x = input

    x = conv2d(x, 64,  (1,x.shape[2]), (1,1), is_training)
    x = conv2d(x, 128, (1,1),          (1,1), is_training)
    x = conv2d(x, 1024,(1,1),          (1,1), is_training)

    x = tf.keras.layers.MaxPool2D(pool_size=(x.shape[1], 1))(x)
    x = tf.squeeze(x, axis=[1,2])

    x = fc(x, 512, is_training)
    x = fc(x, 256, is_training)

    x = tf.keras.layers.Dense(K*K, kernel_initializer="zeros")(x)  # 3*3 matrix

    x = tf.add(x, tf.reshape(tf.eye(K), [-1,]))
    x = tf.reshape(x, [-1,K,K])
    
    return x

def get_backbone(input_pointcloud, num_point, is_training):
    x = tf.expand_dims(input_pointcloud, -1)

    trans = transform_net(x, is_training)
    x = tf.matmul(input_pointcloud, trans)
    
    x = tf.expand_dims(x, -1)
    x = conv2d(x, 64, (1,3), (1,1), is_training)
    x = conv2d(x, 64, (1,1), (1,1), is_training)

    # transform feature here
    # B,512,1,64
    trans = transform_net(x, is_training)
    x = tf.matmul(tf.squeeze(x,[2]), trans)
    x = tf.expand_dims(x, [2])

    x = conv2d(x, 64,   (1,1), (1,1), is_training)
    x = conv2d(x, 128,  (1,1), (1,1), is_training)
    x = conv2d(x, 1024, (1,1), (1,1), is_training)

    x = tf.keras.layers.MaxPool2D(pool_size=(x.shape[1], 1))(x)

    x = tf.squeeze(x, axis=[1,2])

    x = fc(x, 512, is_training)
    x = tf.keras.layers.Dropout(0.3)(x, is_training)

    x = fc(x, 256, is_training)
    x = tf.keras.layers.Dropout(0.3)(x, is_training)

    return x

def get_model_cls(num_point, num_classes, is_training):
    input_pointcloud = tf.keras.Input(shape=(num_point, 3)) # batch_size is optional
    x = get_backbone(input_pointcloud, num_point, is_training)
    x = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs=input_pointcloud, outputs=x)
    return model


def get_model_reg(num_point, num_reg_target, is_training):
    input_pointcloud = tf.keras.Input(shape=(num_point, 3)) # batch_size is optional
    x = get_backbone(input_pointcloud, num_point, is_training)
    
    x = fc(x, 64, is_training)
    x = tf.keras.layers.Dropout(0.3)(x, is_training)

    x = fc(x, 16, is_training)
    x = tf.keras.layers.Dropout(0.3)(x, is_training)


    x = tf.keras.layers.Dense(num_reg_target)(x)

    s = tf.math.sqrt(tf.math.reduce_sum(x*x, axis=-1))
    s = tf.expand_dims(s, -1)
    x = tf.divide(x, s)

    model = tf.keras.Model(inputs=input_pointcloud, outputs=x)
    return model