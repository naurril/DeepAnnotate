
import os
import numpy as np
import json
import math



## read kitti labels.
label_path = '/home/lie/disk640/data/kitti/label_2'
my_label_path = '/home/lie/disk640/data/kitti/sustechpoints_label'
calib_path = "/home/lie/disk640/data/kitti/data_object_calib/training/calib"
pc_path="/home/lie/disk640/data/kitti/velodyne"


def get_inv_matrix(frame):
    with open(os.path.join(calib_path, frame+".txt")) as f:
        lines = f.readlines()
        trans = [x for x in filter(lambda s: s.startswith("Tr_velo_to_cam"), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.split(" ")[1:])]
        matrix = matrix + [0,0,0,1]
        m = np.array(matrix)
        velo_to_cam  = m.reshape([4,4])


        trans = [x for x in filter(lambda s: s.startswith("R0_rect"), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.split(" ")[1:])]        
        m = np.array(matrix).reshape(3,3)
        
        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)
        
        rect = np.concatenate((m, np.expand_dims(np.array([0,0,0,1]), 0)), axis=0)        
        
        
        m = np.matmul(rect, velo_to_cam)

        
        m = np.linalg.inv(m)
        
        return m


files = os.listdir(label_path) 
files.sort()


def make_rotate_matrix(rotation_angle):
    
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],                                
                                [sinval, cosval, 0],
                                [0,0,1],
                                ], 
                                dtype="float32")
    return rotation_matrix


def rotate(points, rotation_matrix):
    return np.dot(points, rotation_matrix)


def crop_obj_round(points, obj):
    #pcfile = os.path.join(pc_path, obj["frame"]+".bin")
    #points = np.fromfile(pcfile, dtype=np.float32).reshape(-1, 4)

    radius = (obj["psr"]["scale"]["x"]*obj["psr"]["scale"]["x"] + obj["psr"]["scale"]["y"]*obj["psr"]["scale"]["y"])/4*2*2 #crop by 2x size
    center = np.array([obj["psr"]["position"]["x"], obj["psr"]["position"]["y"]], dtype="float32")
    dist = points[:,0:2] - center
    
    delta = np.concatenate([center, np.array([0,0],dtype="float32")])
    filtered_points = points[(dist*dist).sum(axis=1)<radius]
    translated_points = filtered_points - delta
    #translated_points.dtype="float32"

    # print("center", center)
    # print("filtered", filtered_points[0:5,:])
    # print("translated", translated_points[0:5,:])
    # print(translated_points.dtype)
    return translated_points, filtered_points


def crop_obj_rect(points, obj_psr):

    radius = (obj_psr["scale"]["x"]*obj_psr["scale"]["x"] + obj_psr["scale"]["y"]*obj_psr["scale"]["y"])/4*2*2 #crop by 2x size
    center = np.array([obj_psr["position"]["x"], obj_psr["position"]["y"], obj_psr["position"]["z"], 0], dtype="float32")
    dist = points[:,0:2] - center[0:2]
    
    points = points[(dist*dist).sum(axis=1)<radius]  # roughly filtered

    points_centered = points - center
    rotated = rotate(points_centered[:,0:3], make_rotate_matrix(obj_psr["rotation"]["z"]))
    
    condition_x = np.abs(rotated[:,0]) < (obj_psr["scale"]["x"]/2+0.3)
    condition_y = np.abs(rotated[:,1]) < (obj_psr["scale"]["y"]/2+0.3)
    #condition_z = rotated[:,2] > -obj_psr["scale"]["z"]/2 + 0.2 # 0.2m

    condition = condition_x & condition_y # & condition_z

    return rotated[condition]  # rotated, centered points

#files = [files[2], files[10]]
def parse_file(fname):
    frame, _ = os.path.splitext(fname)
    print(frame)

  
    inv_m = get_inv_matrix(frame)

    
    with open(os.path.join(label_path, fname)) as f:
        
        pcfile = os.path.join(pc_path, frame+".bin")
        points = np.fromfile(pcfile, dtype=np.float32).reshape(-1, 4)

        lines = f.readlines()
        def parse_one_obj(l):
            words = l.strip().split(" ")
            obj = {}

            pos = np.array([float(words[11]), float(words[12]), float(words[13]), 1]).T
            trans_pos = np.matmul(inv_m, pos)
            #print(trans_pos)

            
            obj_psr = {"scale": {"z":float(words[8]), "x":float(words[9]), "y":float(words[10])},
                            "position": {"x":trans_pos[0], "y":trans_pos[1], "z":trans_pos[2]+float(words[8])/2},
                            #"position": {"x":0, "y":0, "z":0},
                            "rotation": {"x":0, "y":0, "z":math.pi-float(words[14])}}
                            #"rotation": {"x":0, "y":0, "z":0}}
            

            obj["points"] = crop_obj_rect(points, obj_psr)
            obj["frame"] = os.path.splitext(fname)[0]
            obj["obj_id"] = ""
            obj["obj_type"] = words[0]
            obj["psr"] = {"scale": {"z":float(words[8]), "x":float(words[9]), "y":float(words[10])},
                            "position": {"x":0, "y":0, "z":0},
                            "rotation": {"x":0, "y":0, "z":0}}
            
            #obj["points"].tofile("/home/lie/src/SUSTechPoints/data/kitti/pcd/"+frame+"."+ obj["obj_type"] +"."+str(np.random.rand())+".bin")
            return obj
 
        objs = map(parse_one_obj, lines)
        filtered_objs = [x for x in filter(lambda obj: obj["obj_type"]!='DontCare', objs)]
        return filtered_objs

all_objs=[]
for f in files:
    all_objs = all_objs + parse_file(f)

#np.save("all_obs", all_objs)

print("objs num:", len(all_objs))

np.save("all_objs", all_objs)