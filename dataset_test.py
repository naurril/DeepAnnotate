
import data_provider
import numpy as np
import common

def test_data():
    data = data_provider.loadTrainData()
    idx = np.random.randint(data.shape[0])
    
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


        sample = common.sample_one_input_data(obj, common.NUM_POINT, rotate=False)
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

test_data()