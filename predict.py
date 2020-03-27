import tensorflow as tf
import numpy as np
import math
import os
import sys
import model
import common

MODEL_PATH = BASE_DIR+"/log/model.ckpt" #"./log-cls-120-split-train-eval/model.ckpt"  #car only

GPU_INDEX = 0
RESAMPLE_NUM = 5
BATCH_SIZE = RESAMPLE_NUM
def prepare_model():
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = model.placeholder_inputs(BATCH_SIZE, common.NUM_POINT, common.NUM_CLASSES)
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = model.get_model(pointclouds_pl, is_training_pl, common.NUM_CLASSES)
        #loss = model.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()
        
    # Create a session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    print("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           #'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           #'loss': loss
           }
    return sess, ops
    #eval_one_epoch(sess, ops, num_votes)

sess, ops = prepare_model()


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
    
    feed_dict = {ops['pointclouds_pl']: input_data,
                 ops['is_training_pl']: False}
    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)

    #pred_val 5*120, 5*20, 5*20

    pred_cls = np.stack([x for x in map(lambda p: np.argmax(p, 1), pred_val)], axis=1)
    print(pred_cls)
    ret = common.class_to_angle(pred_cls[0])
    ret = ret * (common.CLASS_WEIGHT!=0)
    ret = ret.tolist()
    print(ret)

    return ret


if __name__ == "__main__":
    pred = predict(np.random.random((2048,3)))
    
    print("pred", pred)
    pass