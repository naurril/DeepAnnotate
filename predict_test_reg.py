
import tensorflow as tf
import data_provider
import dataset
import model as M
import numpy as np
import common
import util

util.config_gpu()

d = data_provider.loadEvalData()
test_data = d[0:100]
objs = map(lambda o: common.sample_one_input_data(o, common.NUM_POINT, rotate=True, rotate_value=[0,0, np.pi/3]), test_data)
objs = [x for x in objs]
pcs = np.stack([x['points'] for x in objs], axis=0)
ang = np.array([x['angle'][2:] for x in objs])


weights_path = "../DeepAnnotate/models/da_weights.h5"
#model = tf.keras.models.load_model(model_path)

model = M.get_model_reg(common.NUM_POINT, 2, False)

# model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), 
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=[tf.keras.metrics.sparse_categorical_accuracy, tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)])
#model.summary()

model.load_weights(weights_path)
model.summary()


print(pcs.shape)
print(ang.shape)

pred = model.predict(pcs)

pred_ang = np.arctan2(pred[:,1], pred[:,0])

correct1 = np.sum(np.abs(pred_cls-ang)<1)
correct2 = np.sum(np.abs(pred_cls-ang)<2)
correct3 = np.sum(np.abs(pred_cls-ang)<3)
print(correct1, correct2, correct3, len(ang))

for i,o in enumerate(test_data):
    common.save_to_show(
           "test_predict_"+str(i),
           pcs[i],
           [{
            "obj_type": o["obj_type"],
            "psr": {
                "position": {"x": -objs[i]["translate"][0], 
                             "y": -objs[i]["translate"][1], 
                             "z": -objs[i]["translate"][2]},
                "rotation":{"x": objs[i]["angle"][0], 
                            "y": objs[i]["angle"][1], 
                            "z": pred_cls[i]*3.*np.pi/180.},
                "scale": o["psr"]["scale"],
                },
            "obj_id": o["obj_id"]
           },
           {
            "obj_type": "unknown",
            "psr": {
                "position": {"x": -objs[i]["translate"][0], 
                             "y": -objs[i]["translate"][1], 
                             "z": -objs[i]["translate"][2]},
                "rotation":{"x": objs[i]["angle"][0], 
                            "y": objs[i]["angle"][1], 
                            "z": objs[i]["angle"][2]},
                "scale": o["psr"]["scale"],
                },
            "obj_id": o["obj_id"]
           }

               ])


