
import os
import numpy as np
import json
import math
import glob

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

def box_to_nparray(box):
    return np.array([
        [box["position"]["x"], box["position"]["y"], box["position"]["z"]],
        [box["scale"]["x"], box["scale"]["y"], box["scale"]["z"]],
        [box["rotation"]["x"], box["rotation"]["y"], box["rotation"]["z"]],
    ])
    #box_to_nparray({"rotation":{"x":0, "y":np.pi/3, "z":np.pi/2}, "position":{"x":1,"y":2,"z":3}, "scale":{"x":10,"y":2,"z":5}})

def match_box(box, points):
    center = np.mean(points, axis=0)
    #print(center)
    box_array = box_to_nparray(box['psr'])

    # we translate global coord to local, need to transpose rotate-matrix
    rotate_matrix = euler_angle_to_rotate_matrix(box_array[2,0:3],np.zeros(3))
    rotate_matrix = np.transpose(rotate_matrix)
    center[3] = 0
    center[0:3] = center[0:3] - box_array[0,:]
    center_in_box_coord = np.matmul(rotate_matrix, center)

    # compare with scale/2
    if abs(center_in_box_coord[0]) > box_array[1,0]/2 or abs(center_in_box_coord[1]) > box_array[1,1]/2 or abs(center_in_box_coord[2]) > box_array[1,2]/2:
        return False
    else:
        #print("match", points.shape, center, center_in_box_coord)
        return True


def abs(x):
    return x if x > 0 else -x


def find_best_cluster(box, clusters):
    #print(box)
    #return [match_box(box, c) for c in clusters]
    for i,c in enumerate(clusters):
        if match_box(box, c):
            #print("match cluster", i, "to box", box['obj_id'])
            return i
    return -1


## read kitti labels.
sustechscape_root_path = "/home/lie/fast/code/SUSTechPoints-be/data/sustechscapes-mini-dataset"
label_path = os.path.join(sustechscape_root_path, 'label')
pc_path = os.path.join(sustechscape_root_path, "lidar")

if not os.path.exists("./temp"):
    os.mkdir("./temp")

def pre_cluster_pcd(file, output_folder):
    pre_cluster_exe = "/home/lie/code/pcltest/build/cluster"    
    cmd = "{} {} {}".format(pre_cluster_exe, file, output_folder)
    print(cmd)
    os.system(cmd)



def process_one_frame(frame):    

    with open(os.path.join(label_path, frame+".json")) as tempf:
        label  = json.load(tempf)
    
    
    pcfile = os.path.join(pc_path, frame+".pcd")

    
    temp_output_folder = "./temp/{}".format(frame)
    
    if not os.path.exists(temp_output_folder):
        os.mkdir(temp_output_folder)


    pre_cluster_pcd(pcfile, temp_output_folder)

    cluster_files = glob.glob(temp_output_folder+"/*.bin")
    cluster_files.sort()

    ## note these clusters are already sorted by points number
    clusters = [np.fromfile(c, dtype=np.float32).reshape(-1, 4) for c in cluster_files]
   
    c2bmap = - np.ones(len(clusters), dtype=np.int)

    # for each annotated box, we find a best cluster for them.
    for bi, b in enumerate(label):
        ci = find_best_cluster(b, clusters)
        if ci >= 0:
            if c2bmap[ci] != -1:
                print("conflict! cluster:", ci, "old box:", c2bmap[ci], label[c2bmap[ci]]['obj_id'], "new box:", bi, label[bi]['obj_id'])
            
            c2bmap[ci] = bi
        else:
            print("no match, ", b["obj_type"], b["obj_id"])

    print(c2bmap)
    items = []
    for i,bi in enumerate(c2bmap):

        if bi >= 0:
            item = {
                'points': clusters[i],
                'obj_type':label[bi]['obj_type']
            }
        else:
            item = {
                'points': clusters[i],
                'obj_type':'none'
            }

        items.append(item)

    return items


def main():
    files = os.listdir(pc_path) 
    files.sort()

    all_objs=[]

    for f in files:
        frame, ext = os.path.splitext(f)
        if ext != ".pcd":
            continue
        all_objs = all_objs + process_one_frame(frame)

    np.save("sustechscape_all_objs", all_objs)


main()
