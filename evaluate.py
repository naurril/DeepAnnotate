import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import math
import data_provider
import common
import model


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu

DUMP_DIR = FLAGS.dump_dir


if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


# NUM_CLASSES = 40
# SHAPE_NAMES = [line.rstrip() for line in \
#     open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 


# # ModelNet40 official train/test split
# TRAIN_FILES = data_provider.getDataFiles( \
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = data_provider.getDataFiles(\
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT, common.NUM_CLASSES )
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = model.get_model(pointclouds_pl, is_training_pl, common.NUM_CLASSES)
        loss = model.get_loss(pred, labels_pl, end_points, common.CLASS_WEIGHT)
        
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
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   

def save_to_show(name, points, anno, gt_angle, pred_angle, translate):
    gt_anno = {
                "obj_type":anno["obj_type"],
                "psr":{
                    "position": {
                        "x": 0. - translate[0],
                        "y": 0. - translate[1],
                        "z": 0. - translate[2],
                        },
                    "scale": anno["psr"]["scale"],
                    "rotation": {
                        "x": gt_angle[0],
                        "y": gt_angle[1],
                        "z": gt_angle[2],
                    },
                },
                "obj_id":anno["obj_id"],                    
    }
    pred_anno = {
                "obj_type":"unknown",
                "psr":{
                    "position": {
                        "x": 0. - translate[0],
                        "y": 0. - translate[1],
                        "z": 0. - translate[2],
                        },
                    "scale": anno["psr"]["scale"],
                    "rotation": {
                        "x":pred_angle[0],
                        "y":pred_angle[1],
                        "z":pred_angle[2],
                    },
                },
                "obj_id":anno["obj_id"],                    
    }

    save_anno = [gt_anno, pred_anno]    
    common.save_to_show(name, points, save_anno)



def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = np.zeros(3)
    total_correct_div1 = np.zeros(3)
    total_correct_div2 = np.zeros(3)
    total_seen = 0
    loss_sum = 0
    #total_seen_class = [0 for _ in range(NUM_CLASSES)]
    #total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    if True: #for fn in range(len(TEST_FILES)):
        #log_string('----' + str(fn) + '-----')
        data = data_provider.loadEvalData()
        #current_data = current_data[:,0:NUM_POINT,:]
        #data = data_provider.shuffle_data(data)
        
        num_batches = data.shape[0] // BATCH_SIZE

        file_size = data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data = [d for d in map(lambda o: common.sample_one_input_data(o,common.NUM_POINT), data[start_idx:end_idx])]
            input_data = np.stack([i for i in map(lambda o: o["points"], batch_data)], axis=0)
            input_angle_cls = np.stack([i for i in map(lambda o: common.angle_to_class(o["angle"]), batch_data)], axis=0)#B*3
            input_angle = np.stack([i for i in map(lambda o: o["angle"], batch_data)], axis=0)#, dtype="float32"),

            feed_dict = {ops['pointclouds_pl']: input_data,
                         ops['labels_pl']: input_angle_cls,
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            total_seen += BATCH_SIZE

            pred_cls = np.stack([x for x in map(lambda p: np.argmax(p,1), pred_val)], axis=0).T
            if True:
                print(pred_cls-input_angle_cls)

                
                total_correct      += np.sum(pred_cls == input_angle_cls, axis=0)
                total_correct_div1 += np.sum(np.abs(pred_cls - input_angle_cls)<=1, axis=0)
                total_correct_div2 += np.sum(np.abs(pred_cls - input_angle_cls)<=2, axis=0)

                for i in range(input_data.shape[0]): # by batch index
                    points = input_data[i]
                    pred_angle = common.class_to_angle(pred_cls[i])
                    gt_angle = input_angle[i]
                    obj_info = data[start_idx+i]
                    translate = batch_data[i]["translate"]
                    save_to_show("eval"+str(start_idx+i), points, obj_info, gt_angle, pred_angle, translate)


            loss_sum += (loss_val*BATCH_SIZE)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    print('eval accuracy', total_correct/float(total_seen))
    print('eval accuracy div1', total_correct_div1/float(total_seen))
    print('eval accuracy div2', total_correct_div2/float(total_seen))
    #log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))



if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
