
import tensorflow as tf
import common


def get_dataset(data):
    #point_clouds = [x for x in map(lambda o: o["points"], data)]
    index_data = tf.data.Dataset.from_tensor_slices([x for x in range(len(data))])
    index_data = index_data.shuffle(buffer_size=len(data))

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