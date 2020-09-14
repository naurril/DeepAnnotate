
import tensorflow as tf
import common
import data_provider


"""
dataset pipeline

1. raw-dataset         N*3
2. rotate/crop bottom/sample, 512*3, xyz angle
3.1 for classification model, z-angle->category
3.2 for regression            z-angle->(cos,sin)

"""


def get_dataset(data):
    #point_clouds = [x for x in map(lambda o: o["points"], data)]
    index_data = tf.data.Dataset.from_tensor_slices([x for x in range(len(data))])
    index_data = index_data.shuffle(buffer_size=len(data))

    def sample(idx):
        sample = common.sample_one_input_data(data[idx], common.NUM_POINT)
        #print(sample["angle"])
        #angle_class = common.angle_to_class(sample["angle"])
        #z_angle = angle_class[2:]

        return sample["points"], sample["angle"] #only z angle, IMPORTANT: don't use scalar value

    def tf_rotate_object(obj_idx):
        [points, angle] = tf.py_function(sample, [obj_idx], [tf.float32, tf.float32])
        points.set_shape((common.NUM_POINT, 3))
        angle.set_shape((3,))
        return points, angle

    input_data = index_data.map(tf_rotate_object)
    return input_data


def get_cls_dataset_x(data):  
    "get tf.dataset without label"
    #point_clouds = [x for x in map(lambda o: o["points"], data)]
    index_data = tf.data.Dataset.from_tensor_slices([x for x in range(len(data))])
    index_data = index_data.shuffle(buffer_size=len(data))

    def sample(idx):
        sample = common.sample_one_input_data(data[idx], common.NUM_POINT, rotate=False)
        return sample["points"]

    def tf_rotate_object(obj_idx):
        [points, _] = tf.py_function(sample, [obj_idx], [tf.float32, tf.float32])
        points.set_shape((common.NUM_POINT, 3))        
        return points

    input_data = index_data.map(tf_rotate_object)
    return input_data




# classifcation dataset

def tf_angle_to_cls(points, angle):
    ang = angle
    ang_cls = tf.cast((ang - common.CLASS_LOWEST_RAD)/common.CLASS_GAP, tf.int64)
    ang_cls = ang_cls[2:]
    return points, ang_cls

def get_cls_train_dataset():
    data = data_provider.loadTrainData()
    data = get_dataset(data)    
    data = data.map(tf_angle_to_cls)
    return data

def get_cls_eval_dataset():
    data = data_provider.loadEvalData()
    data = get_dataset(data)
    data = data.map(tf_angle_to_cls)
    return data


# regression dataset

def tf_angle_to_reg(points, angle):
    angle = angle[2]
    target = [tf.cos(angle), tf.sin(angle)]
    return points, target

def get_reg_train_dataset():
    data = data_provider.loadTrainData()
    data = get_dataset(data)    
    #data = data.map(tf_angle_to_reg)
    return data

def get_reg_eval_dataset():
    data = data_provider.loadEvalData()
    data = get_dataset(data)
    data = data.map(tf_angle_to_reg)
    return data