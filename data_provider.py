import os
import numpy as np

POINTS_NUMBER = 512

all_objs = np.load("all_objs.npy", allow_pickle=True)
#valid_obj_flag =[i for i in map(lambda o: (o["points"].shape[0]>=POINTS_NUMBER) and (o["obj_type"]=="Car"),  all_objs)]
valid_obj_flag =[i for i in map(lambda o: (o["points"].shape[0]>=POINTS_NUMBER/2) ,  all_objs)]
valid_objs = all_objs[valid_obj_flag]
train_obj_num = int(len(valid_objs) * 0.8)


def shuffle_data(data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx]


def sample_one_obj(points, num):
    idx = np.arange(points.shape[0])
    np.random.shuffle(idx)
    return points[idx[0:num]]


def loadTrainData():
    return np.array([x for x in map(lambda o: {"points":o["points"][:,0:3], 
                                               "angle": o["psr"]["rotation"]["z"],
                                               "rotation": o["psr"]["rotation"],
                                               "psr": o["psr"], # used for evaluation
                                               "obj_type": o["obj_type"], #used for evaluation
                                               "obj_id": o["obj_id"]
                                               }, valid_objs[0:train_obj_num])])


def loadEvalData():
    return np.array([x for x in map(lambda o: {"points":o["points"][:,0:3], 
                                               "angle": o["psr"]["rotation"]["z"],
                                               "rotation": o["psr"]["rotation"],
                                               "psr": o["psr"], # used for evaluation
                                               "obj_type": o["obj_type"], #used for evaluation
                                               "obj_id": o["obj_id"]
                                               }, valid_objs[train_obj_num:])])

