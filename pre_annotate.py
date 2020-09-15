
import os
import tensorflow as tf
import data_provider
import dataset
import model as M
import numpy as np
import common
import util
import glob
import math
import json

util.config_gpu()


# weights_path = "../DeepAnnotate/da_rp_weights.h5"
# #filter_model = tf.keras.filter_models.load_filter_model(filter_model_path)

# filter_model = M.get_filter_model_rp_discrimination(common.NUM_POINT, 2, False)

# # filter_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
# #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# #             metrics=[tf.keras.metrics.sparse_categorical_accuracy, tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)])
# #filter_model.summary()

# filter_model.load_weights(weights_path)
# filter_model.summary()



filter_model_file = "./deepannotate_rp_discrimination.back.h5"
filter_model = tf.keras.models.load_model(filter_model_file)
filter_model.summary()


rotation_model_file = "../SUSTechPoints-be/algos/models/deep_annotation_inference.h5"
rotation_model = tf.keras.models.load_model(rotation_model_file)
rotation_model.summary()

NUM_POINT = 512

def cluster_points(pcfile):
    def pre_cluster_pcd(file, output_folder):
        pre_cluster_exe = "/home/lie/code/pcltest/build/cluster"    
        cmd = "{} {} {}".format(pre_cluster_exe, file, output_folder)
        print(cmd)
        os.system(cmd)

    temp_output_folder = "./temppcd"
    if os.path.exists(temp_output_folder):
        os.system("rm {}/*".format(temp_output_folder))
    else:
        os.mkdir(temp_output_folder)

    pre_cluster_pcd(pcdfile, temp_output_folder)

    cluster_files = glob.glob(temp_output_folder+"/*.bin")
    cluster_files.sort()

    ## note these clusters are already sorted by points number
    clusters = [np.fromfile(c, dtype=np.float32).reshape(-1, 4)[:,0:3] for c in cluster_files]
    return clusters,cluster_files

def euler_angle_to_rotate_matrix(eu, t):
    theta = eu
    #Calculate rotation about x axis
    R_x = np.array([
        [1,       0,              0],
        [0,       math.cos(theta[0]),   -math.sin(theta[0])],
        [0,       math.sin(theta[0]),   math.cos(theta[0])]
    ])

    #Calculate rotation about y axis
    R_y = np.array([
        [math.cos(theta[1]),      0,      math.sin(theta[1])],
        [0,                       1,      0],
        [-math.sin(theta[1]),     0,      math.cos(theta[1])]
    ])

    #Calculate rotation about z axis
    R_z = np.array([
        [math.cos(theta[2]),    -math.sin(theta[2]),      0],
        [math.sin(theta[2]),    math.cos(theta[2]),       0],
        [0,               0,                  1]])

    R = np.matmul(R_x, np.matmul(R_y, R_z))

    t = t.reshape([-1,1])
    R = np.concatenate([R,t], axis=-1)
    R = np.concatenate([R, np.array([0,0,0,1]).reshape([1,-1])], axis=0)
    return R


def sample_one_obj(points, num):
    centroid = np.mean(points, axis=0)
    points = points - centroid

    if points.shape[0]>num:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        points =  points[idx[0:num]]
    else:
        sample_idx = np.random.randint(0, high = points.shape[0], size=num - points.shape[0])
        padding = points[sample_idx]
        points = np.concatenate([points, padding], axis=0)

    return points

# return true/false list
def filter_candidate_objects(clusters):

    ## all clusters stacked into a batch
    input_cluster_points = np.stack([sample_one_obj(p, NUM_POINT) for p in clusters], axis=0)

    pred_val = filter_model.predict(input_cluster_points)
    #print(pred_val)
    prob = np.exp(pred_val)/np.sum(np.exp(pred_val),axis=1,keepdims=True)

    #print(prob)
    #pred_cls = np.argmax(prob, axis=1)
    pred_cls = prob[:,1]>0.5
    return pred_cls

def decide_obj_rotation(objs):
    input_data = np.stack([sample_one_obj(o, NUM_POINT) for o in objs], axis=0)
    pred_val = rotation_model.predict(input_data)
    pred_cls = np.argmax(pred_val, axis=-1)
    
    ret = (pred_cls*3+1.5)*np.pi/180.0  #only z-axis rotation is predicted
    
    return ret

def calculate_box_dimension(objs, rotation):

    def calc_one_box(obj, rot):
        #print(obj.shape, rot)
        rot = np.array([0,0,rot])
        centroid = np.mean(obj, axis=0)
        obj = obj - centroid
        
        trans_mat = euler_angle_to_rotate_matrix(rot, np.zeros(3))[:3,:3]
        relative_position = np.matmul(obj, trans_mat)

        pmin = np.min(relative_position, axis=0)        
        pmax = np.max(relative_position, axis=0)
        pmin[2]  = pmin[2] - 0.2 # remember 0.2 was removed, as ground

        box_dim = pmax-pmin
        box_center_delta = box_dim/2 + pmin
        #center delta shoulb be translated to global coord
        box_center_delta = np.matmul(trans_mat, box_center_delta)

        box_center = box_center_delta + centroid
        
        return np.stack([box_center, box_dim, rot],axis=0)

            
    return [calc_one_box(obj, rot) for obj,rot in zip(objs, rotation)]



## main func
def pre_annotate(pcdfile):
    clusters,cluster_files = cluster_points(pcdfile)
    cand_ind = filter_candidate_objects(clusters)

    # positive_files = np.array(cluster_files)[cand_ind]
    # positive_files = [x for x in map(lambda f: f.replace(".bin",".pcd"), list(positive_files))]

    # outstr = ""
    # for f in positive_files:
    #     outstr = outstr + " " + f

    # print(outstr)


    # calculate box
    cand_clusters = np.array(clusters)[cand_ind]
    cand_rotation = decide_obj_rotation(cand_clusters)

    # print(cand_rotation)
    boxes = calculate_box_dimension(cand_clusters, cand_rotation)
    print(boxes)
    return boxes


def translate_np_to_json(boxes):
    def trans_one_box(box):
        return {
            'obj_type': 'Car',
            'psr': {
                'position': {
                    'x': box[0,0],
                    'y': box[0,1],
                    'z': box[0,2],
                },
                'scale': {
                    'x': box[1,0],
                    'y': box[1,1],
                    'z': box[1,2],
                },
                'rotation': {
                    'x': box[2,0],
                    'y': box[2,1],
                    'z': box[2,2],
                }
            },
            'obj_id': '',            
        }
    
    return [trans_one_box(b) for b in boxes]

pcdfile = "/home/lie/fast/code/SUSTechPoints-be/data/2020-07-12-15-30-24/lidar/946657013.000444000.pcd"
boxes = pre_annotate(pcdfile)
boxes_json = translate_np_to_json(boxes)
with open("/home/lie/fast/code/SUSTechPoints-be/data/2020-07-12-15-30-24/label/946657013.000444000.json", 'w') as f:
    json.dump(boxes_json, f)