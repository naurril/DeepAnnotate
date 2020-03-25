import argparse
import math
import h5py
import numpy as np
#import tensorflow as tf
import importlib
import os
import sys
import data_provider
import common
import model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOG_DIR = FLAGS.log_dir


if not os.path.exists(LOG_DIR): 
    os.mkdir(LOG_DIR)


# save_src_files
os.system('cp *.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')




BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

use_regression = False

# ModelNet40 official train/test split
# TRAIN_FILES = data_provider.getDataFiles( \
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = data_provider.getDataFiles(\
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT, common.NUM_CLASSES,  regression=use_regression)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = model.get_model(pointclouds_pl, is_training_pl, common.NUM_CLASSES, bn_decay=bn_decay, regression=use_regression)
            loss = model.get_loss(pred, labels_pl, end_points, common.CLASS_WEIGHT, reg_weight=0.001, regression=use_regression)
            tf.summary.scalar('loss', loss)
            

            if use_regression:
                correct = tf.equal(tf.to_int64(pred*180./np.pi/common.NUM_CLASSES), tf.to_int64(labels_pl*180./np.pi/common.NUM_CLASSES))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                tf.summary.scalar('accuracy', accuracy)  #accuracy of current batch
            else:
                pred_cls = [x for x in map(lambda p: tf.argmax(p, axis=1), pred)]
                pred_cls = tf.stack(pred_cls, axis=1)  # N*3
                correct = tf.equal(pred_cls, tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32), axis=0) / float(BATCH_SIZE)
                for i in range(accuracy.shape[0]):
                    tf.summary.scalar('accuracy_'+str(i), accuracy[i])  #accuracy of current batch

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if (epoch+1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def test_data():
    data = data_provider.loadTrainData()
    #idx = np.random.randint(data.shape[0])
    num = 10
    if num > data.shape[0]:
        num = data.shape[0]
    for i in range(10):        
        obj = data[i]

        print(obj["points"].shape)

        common.save_to_show(
           "test_data_"+str(i)+"_org",
           obj["points"], 
           [{
            "obj_type": obj["obj_type"],
            "psr": obj["psr"],
            "obj_id": obj["obj_id"]
           }])


        sample = common.sample_one_input_data(obj, common.NUM_POINT)
        print(sample["points"].shape)
        common.save_to_show(
           "test_data_"+str(i),
           sample["points"], 
           [{
            "obj_type": obj["obj_type"],
            "psr": {
                "position": {"x": -sample["translate"][0], 
                             "y": -sample["translate"][1], 
                             "z": -sample["translate"][2]},
                "rotation":{"x": sample["angle"][0], 
                            "y": sample["angle"][1], 
                            "z": sample["angle"][2]},
                "scale": obj["psr"]["scale"],
                },
            "obj_id": obj["obj_id"]
           }])



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    # train_file_idxs = np.arange(0, len(TRAIN_FILES))
    # np.random.shuffle(train_file_idxs)
    
    #for fn in range(len(TRAIN_FILES)):
    if True: # we have only 1 file
        data = data_provider.loadTrainData()
        #current_data = current_data[:,0:NUM_POINT,:]
        data = data_provider.shuffle_data(data)
        
        num_batches = data.shape[0] // BATCH_SIZE
        
        total_correct = np.zeros(3)
        total_seen = 0
        loss_sum = 0
        total_squared_diff = 0.0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            
            batch_data = [d for d in map(lambda o: common.sample_one_input_data(o,common.NUM_POINT), data[start_idx:end_idx])]
            input_data = np.stack([i for i in map(lambda o: o["points"], batch_data)], axis=0)
            input_angle_cls = np.stack([i for i in map(lambda o: common.angle_to_class(o["angle"]), batch_data)], axis=0)#B*3
                          


            # Augment batched point clouds by rotation and jittering
            #rotated_data = current_data[start_idx:end_idx, :, :] #data_provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            #jittered_data = data_provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: input_data,
                         ops['labels_pl']: input_angle_cls,
                         ops['is_training_pl']: is_training,                         
                         }
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            
            pred_cls = np.stack([x for x in map(lambda p: np.argmax(p, 1), pred_val)], axis=1)  #B*3
            
            correct = np.sum(pred_cls == input_angle_cls, axis=0)

            total_correct += correct

            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('mean squared diff: %f' % (total_squared_diff / float(total_seen)))
        log_string('accuracy: %f, %f, %f' % (total_correct[0] / float(total_seen), total_correct[1] / float(total_seen), total_correct[2] / float(total_seen)))

        
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

            feed_dict = {ops['pointclouds_pl']: input_data,
                         ops['labels_pl']: input_angle_cls,
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            
            total_seen += BATCH_SIZE

            pred_cls = np.stack([x for x in map(lambda p: np.argmax(p, 1), pred_val)], axis=1)  #B*3
            
            total_correct += np.sum(np.abs(pred_cls - input_angle_cls)<1, axis=0)
            total_correct_div1 += np.sum(np.abs(pred_cls - input_angle_cls)<=1, axis=0)
            total_correct_div2 += np.sum(np.abs(pred_cls - input_angle_cls)<=2, axis=0)

            loss_sum += loss_val
            
            
                
            loss_sum += (loss_val*BATCH_SIZE)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    print('eval accuracy:', total_correct/float(total_seen))
    print('eval accuracy div1', total_correct_div1/float(total_seen))
    print('eval accuracy div2', total_correct_div2/float(total_seen))
    #log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))


if __name__ == "__main__":
    train()
    #LOG_FOUT.close()

    # test data augmentation
    #test_data()
