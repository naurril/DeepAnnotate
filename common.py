import numpy as np
import math
import json

NUM_POINT = 512
NUM_CLASSES      = np.array([20, 20, 120])
CLASS_WEIGHT     = np.array([0,0,1])
_CLASS_LOWEST_DEGREE = np.array([-5, -5, 0])
CLASS_LOWEST_RAD    = _CLASS_LOWEST_DEGREE * np.math.pi/180
_CLASS_COVERED_DEGREES      = np.array([10, 10, 360])
CLASS_COVERED_RAD      = _CLASS_COVERED_DEGREES * np.math.pi/180
CLASS_GAP = CLASS_COVERED_RAD/NUM_CLASSES


def angle_to_class(ang):
    ang = np.array(ang)
    return np.int32((ang - CLASS_LOWEST_RAD)/CLASS_GAP)

def class_to_angle(cls):
    return cls * CLASS_GAP + CLASS_LOWEST_RAD


def rotate_one_obj(points, rotation=None):

    if not rotation:
        rotation = (np.random.uniform(size=3) + CLASS_LOWEST_RAD/CLASS_COVERED_RAD) * CLASS_COVERED_RAD
    #print(angle_z, angle_y, angle_x)
    #angle_z, angle_y, angle_x = [0, 0, np.pi/6]
    # we rotate only yaw angle
    cosval = np.cos(rotation[2])
    sinval = np.sin(rotation[2])
    rotation_matrix_z = np.array([[cosval, -sinval, 0],                                
                                  [sinval, cosval, 0],
                                  [0,      0,      1]], 
                                dtype="float32")

    cosval = np.cos(rotation[1])
    sinval = np.sin(rotation[1])
    rotation_matrix_y = np.array([[cosval, 0, sinval],
                                  [0,      1,       0],
                                  [-sinval, 0,  cosval],
                                ], 
                                dtype="float32")

    cosval = np.cos(rotation[0])
    sinval = np.sin(rotation[0])
    rotation_matrix_x = np.array([[1,      0,      0],
                                  [0,   cosval, -sinval],                                
                                  [0,   sinval, cosval],
                                ], 
                                dtype="float32")

    rotation_matrix = np.matmul(rotation_matrix_x, np.matmul(rotation_matrix_y, rotation_matrix_z)).T
    
    return np.dot(points, rotation_matrix), rotation


def crop_out_bottom(points, threshold=0.3):
    min_z = np.min(points[:,2])
    condition = (points[:, 2] - min_z) > threshold
    points = points[condition]
    return points

def sample_one_input_data(obj, num_points, rotate=True, rotate_value=None, translate=True, crop=True, cut_bottom=True):
    points = obj["points"]
    label_rotation = np.array([obj["rotation"]["x"], obj["rotation"]["y"], obj["rotation"]["z"]])

    # 1 rotate
    if rotate:
        points, rotation = rotate_one_obj(points, rotate_value)
        #points = provider.jitter_point_cloud(points)
        label_rotation = rotation + label_rotation
    

    if cut_bottom:
        # 1.1 crop out bottom part
        points = crop_out_bottom(points, 0.3)


    # 2 translate
    if translate:
        # the center  is  actually (0,0,x), although psr.position hold the original value
        
        scale = np.array([obj["psr"]["scale"]["x"], obj["psr"]["scale"]["y"], obj["psr"]["scale"]["z"]], dtype=np.float32)
        translate = (np.random.random(3) - 0.5)*0.4 * scale
        translate = np.float32(translate)

        # 3 crop    
        points = points - translate

    if crop:
        # drop points with distance to center greater than 1.5*scale 
        condition = np.sum(np.square(points[:,0:2]), axis=1) <  (scale[0]*scale[0] + scale[1]*scale[1]) * 2.25/4
        points = points[condition]

    #print("points shape", points.shape)
    # 4 sample or padding
    # print(points.shape)
    if points.shape[0]>num_points:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        points =  points[idx[0:num_points]]
    else:
        sample_idx = np.random.randint(0, high = points.shape[0], size=num_points - points.shape[0])
        padding = points[sample_idx]
        points = np.concatenate([points, padding], axis=0)

    # jitter

    # center again

    # normalize

    # return
    return {
        "points": points, 
        "angle": label_rotation.tolist(),
        "translate": translate.tolist()
        }




def sample_one_rp_data(obj, num_points, rotate=True, rotate_value=None, translate=True, crop=True):
    points = obj["points"][:,:3] #don't use intensity
    
    #centroid = np.mean(points, axis=0)
    # centroid[3] = 0.0  #intensity, keep original value?

    #points = points - centroid

    # 1 rotate
    if rotate:
        points, rotation = rotate_one_obj(points, rotate_value)
        #points = provider.jitter_point_cloud(points)
        #label_rotation = rotation + label_rotation
    
    #points = points + centroid

    #print("points shape", points.shape)
    # 4 sample or padding
    # print(points.shape)
    if points.shape[0]>num_points:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        points =  points[idx[0:num_points]]
    else:
        sample_idx = np.random.randint(0, high = points.shape[0], size=num_points - points.shape[0])
        padding = points[sample_idx]
        points = np.concatenate([points, padding], axis=0)

    # jitter

    # center again

    # normalize

    clz = 1 if obj["obj_type"] != 'none' else 0
    # return
    return {
        "points": points, 
        "clz": [clz]
        }



viewer_path = "../SUSTechPOINTS/data/kitti_eval"

def save_to_show(name, points, anno):    
    padding = np.zeros([points.shape[0], 1], dtype=np.float32)   # pad to N*4
    points = np.concatenate([points, padding], axis=-1)
    points.tofile(viewer_path + "/pcd/"+name+".bin")
    #print(points.dtype)
    
    with open(viewer_path + "/label/"+name+".json", 'w') as outfile:
                json.dump(anno, outfile)