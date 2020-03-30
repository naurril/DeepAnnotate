

import numpy as np
import tensorflow as tf
import common
import util
import model as M

util.config_gpu()

RESAMPLE_NUM = 20

weights_path = "../DeepAnnotate/models/da_weights.h5"
model = M.get_model_tf2(common.NUM_POINT, common.NUM_CLASSES[2], False)
model.load_weights(weights_path)
model.summary()


def sample_one_obj(points, num):
    if points.shape[0] < common.NUM_POINT:
        return np.concatenate([points, np.zeros((common.NUM_POINT-points.shape[0], 3), dtype=np.float32)], axis=0)
    else:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        return points[idx[0:num]]

def predict(points):
    points = np.array(points).reshape((-1,3))
    input_data = np.stack([x for x in map(lambda x: sample_one_obj(points, common.NUM_POINT), range(RESAMPLE_NUM))], axis=0)
    pred_val = model.predict(input_data)
    pred_cls = np.argmax(pred_val, axis=-1)
    print(pred_cls)
    ret = common.class_to_angle(pred_cls[0])
    ret = ret * (common.CLASS_WEIGHT!=0)
    ret = ret.tolist()
    print(ret)

    return ret


if __name__ == "__main__":

    predict(np.random.random([1000,3]))