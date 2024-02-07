import pandas as pd
import numpy as np
from os import listdir

class DatasetTraining:
    def __init__(self, train_path_folder, val_path_folder, index_col, feature_name, sliding_window_size, scaling_method="normalization", ahead_steps=1):
        if scaling_method != "normalization":
            raise ValueError("Invalid scaling method. Please use 'normalization'")

        self.train_path_folder = train_path_folder
        self.val_path_folder = val_path_folder
        self.index_col = index_col
        self.feature_name = feature_name
        self.sliding_window_size = sliding_window_size
        self.scaling_method = scaling_method
        self.ahead_steps = ahead_steps

        self.train_datasets = []
        self.val_datasets = []

        self.train_datasets_scaled = []
        self.val_datasets_scaled = []

        self.test_dataset = None

    def preprocess_data(self):
        self.load_data()
        self.scale_values()
        self.create_training_validation_data()

    def load_data(self):
        print("Loading training data...", end=" ")

        for f in listdir(self.train_path_folder):
            if not f.endswith(".csv"):
                continue
            path = self.train_path_folder + "/" + f
            data = pd.read_csv(path, index_col=self.index_col)
            data.index = pd.to_datetime(data.index)
            if data.empty:
                continue
            self.train_datasets.append(data.copy())
        print("Done!")

        print("Loading validation data...", end=" ")
        for f in listdir(self.val_path_folder):
            if not f.endswith(".csv"):
                continue
            path = self.val_path_folder + "/" + f
            data = pd.read_csv(path, index_col=self.index_col)
            data.index = pd.to_datetime(data.index)
            if data.empty:
                continue
            self.val_datasets.append(data.copy())
        print("Done")
    
    def scale_values(self, min_value=30):
        print("Computing mean ON cycle consumption...", end=" ")
        max_values = []
        for d in self.train_datasets:
            tmp = d.loc[d[self.feature_name] > min_value]
            if not tmp.empty:
                max_values.append(tmp[self.feature_name].mean())
        self.max_v = np.mean(max_values)
        self.min_v = 0
        print("Done: {}!".format(self.max_v))

        print("Scaling feature...", end=" ")
        for d in self.train_datasets:
            feature_values = d[self.feature_name].values
            scaled_dataset = pd.DataFrame({self.feature_name: self.__normalize_values(feature_values) if self.scaling_method == "normalization" else self.__standardize_values(feature_values)},
                                          index=d.index.values)
            self.train_datasets_scaled.append(scaled_dataset.copy())
        
        for d in self.val_datasets:
            feature_values = d[self.feature_name].values
            scaled_dataset = pd.DataFrame({self.feature_name: self.__normalize_values(feature_values) if self.scaling_method == "normalization" else self.__standardize_values(feature_values)},
                                          index=d.index.values)
            self.val_datasets_scaled.append(scaled_dataset.copy())
        print("Done!")
    
    def create_training_validation_data(self):
        print("Creating training data...", end=" ")

        # create sliding windows
        self.X_train, self.X_train_forecasting, self.Y_train_forecasting = [], [], []
        for d in self.train_datasets_scaled:
            s = d[self.feature_name].values
            x_train, x_train_forecasting, y_train_forecasting = self.__create_sliding_windows(s)
            self.X_train.extend(x_train)
            self.X_train_forecasting.extend(x_train_forecasting)
            self.Y_train_forecasting.extend(y_train_forecasting)
        
        self.X_train = np.array(self.X_train).reshape(-1, self.sliding_window_size, 1)
        self.X_train_forecasting = np.array(self.X_train_forecasting).reshape(-1, self.sliding_window_size, 1)
        self.Y_train_forecasting = np.array(self.Y_train_forecasting).reshape(-1, self.ahead_steps)

        print("Done!")

        print("Creating validation data...", end=" ")

        # create sliding windows
        self.X_val, self.X_val_forecasting, self.Y_val_forecasting = [], [], []
        for d in self.val_datasets_scaled:
            s = d[self.feature_name].values
            x_val, x_val_forecasting, y_val_forecasting = self.__create_sliding_windows(s)
            self.X_val.extend(x_val)
            self.X_val_forecasting.extend(x_val_forecasting)
            self.Y_val_forecasting.extend(y_val_forecasting)
        
        self.X_val = np.array(self.X_val).reshape(-1, self.sliding_window_size, 1)
        self.X_val_forecasting = np.array(self.X_val_forecasting).reshape(-1, self.sliding_window_size, 1)
        self.Y_val_forecasting = np.array(self.Y_val_forecasting).reshape(-1, self.ahead_steps)

        print("Done!")
    
    def get_test_data(self, path):
        data = pd.read_csv(path, index_col=self.index_col)
        data.index = pd.to_datetime(data.index)

        data[self.feature_name] = self.__normalize_values(data[self.feature_name].values)

        X_test, X_test_forecasting, Y_test_forecasting = self.__create_sliding_windows(data[self.feature_name].values)

        X_test, X_test_forecasting, Y_test_forecasting = np.array(X_test).reshape(-1, self.sliding_window_size, 1), np.array(X_test_forecasting).reshape(-1, self.sliding_window_size, 1), np.array(Y_test_forecasting).reshape(-1, self.ahead_steps)
        return X_test, X_test_forecasting, Y_test_forecasting, data.copy()
    
    def __create_sliding_windows(self, x):
        if len(x) <= self.sliding_window_size:
            return [], [], []
        X_train, X_train_forecast, Y_train_forecast = [], [], []
        
        for i in range(0, (len(x)-self.sliding_window_size)+1):
            X_train.append(x[i:i+self.sliding_window_size])

            if i+self.sliding_window_size+self.ahead_steps > len(x):
                continue
            X_train_forecast.append(x[i:i+self.sliding_window_size])
            Y_train_forecast.append(x[i+self.sliding_window_size:i+self.sliding_window_size+self.ahead_steps])

        return X_train, X_train_forecast, Y_train_forecast
    
    def __normalize_values(self, x):
        return (x-self.min_v) / (self.max_v-self.min_v)
    
    def _inverse_normalize_values(self, scaled_x):
        return scaled_x*(self.max_v-self.min_v) + self.min_v
    
    def __standardize_values(self, x):
        # TODO
        return x
