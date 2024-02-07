from dataset import DatasetTraining
from model import CustomModel
from evaluation import EvaluateModel
import os
import sys


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    args = sys.argv
    DATASET = args[1]
    TRAINING_SIZE = args[2]
    PERIOD = args[3]
    WINDOW = int(args[4])

    MODEL_NAME = args[5]
    MODEL_SHORT_NAME = args[6]
    
    
    AHEAD_STEPS = int(args[7])
        
    validation_folder = f"../data/{DATASET}/val/clean_data"
    testing_file = f"../data/{DATASET}/val/val_{DATASET}.csv"
    
    training_folder = f"../data/{DATASET}/train/clean_data/{TRAINING_SIZE}"
        

    weights_path = f"../trained_models/EIA2022_CERTH_{DATASET}_{MODEL_SHORT_NAME}_{TRAINING_SIZE}_{PERIOD}.h5"

    results_output_path = f"../predictions_val/EIA2022_CERTH_{DATASET}_{MODEL_SHORT_NAME}_{TRAINING_SIZE}_{PERIOD}.csv"

    dataset = DatasetTraining(training_folder, 
                  validation_folder,
                  index_col="ctime",
                  feature_name="device_consumption",
                  sliding_window_size=WINDOW,
                  ahead_steps=AHEAD_STEPS)
    
    dataset.preprocess_data()

    model = CustomModel(MODEL_NAME,
                        dataset,
                        model_weights_path=weights_path)
    
    model.get_validation_performance()

    evaluator = EvaluateModel(model, testing_file)

    evaluator.generate_predictions(results_output_path)
