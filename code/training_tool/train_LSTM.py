from dataset import DatasetTraining
from model import CustomModel
import os
import json
import sys

def my_scheduler(epoch, learning_rate):
    if epoch == 0:
        return 0.001
    if learning_rate < 0.0001:
        return 0.0001
    if epoch % 5 == 0:
        return learning_rate/2
    return learning_rate

if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = sys.argv
    dataset_name = args[1]
    training_time = args[2]
    period_time = args[3]
    window_size = int(args[4])

    
    dataset = DatasetTraining(f"../data/{dataset_name}/train/clean_data/{training_time}", 
                              f"../data/{dataset_name}/val/clean_data",
                              index_col="ctime",
                              feature_name="device_consumption",
                              sliding_window_size=window_size,
                              ahead_steps=1)
    dataset.prepocess_data()
    
    print(dataset.X_train_forecasting.shape, dataset.X_val_forecasting.shape)

    model = CustomModel("LSTM",
                        dataset)

    model.train(epochs=500, batch_size=64, patience=30, scheduler=my_scheduler)
    
    with open(f'../trained_models/EIA2022_CERTH_{dataset_name}_LSTM_{training_time}_{period_time}_loss.json', 'w') as f:
        json.dump(model.log, f, indent=4)
        f.close()
        
    model.model.save(f"../trained_models/EIA2022_CERTH_{dataset_name}_LSTM_{training_time}_{period_time}.h5")