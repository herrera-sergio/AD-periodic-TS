from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, GRU, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping, LearningRateScheduler

import numpy as np
import matplotlib.pyplot as plt

from dataset import DatasetTraining

class CustomModel:

    __available_models = {'LSTM', 'GRU', 'LSTM-multisteps', 'GRU-multisteps', 'LSTM-Autoencoder', 'GRU-Autoencoder'}

    __forecasting_models = {'LSTM', 'GRU', 'LSTM-multisteps', 'GRU-multisteps'}
    
    def __init__(self,
                model_name,
                dataset_training,
                model_weights_path=None):
        
        if model_name not in self.__available_models:
            raise Exception("Invalid model. Available models: {}".format(self.__available_models))
        if not isinstance(dataset_training, DatasetTraining):
            raise TypeError("Invalid 'dataset_training' type. Use 'DatasetTraining' class.")
        
        self.dataset = dataset_training
        self.model_name = model_name
        if model_weights_path is None:
            self.model = self._create_model()
        else:
            self.model = load_model(model_weights_path)
        self.log = {"loss": [], "val_loss": []}
    
    def train(self, epochs, batch_size, patience=None, scheduler=None):
        callbacks = []
        if patience is not None:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=patience))
        if scheduler is not None:
            callbacks.append(LearningRateScheduler(scheduler, verbose=1))
        if len(callbacks) == 0:
            callbacks = None
        if self.model_name in self.__forecasting_models:
            log = self.model.fit(self.dataset.X_train_forecasting, 
                                        self.dataset.Y_train_forecasting,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(self.dataset.X_val_forecasting, self.dataset.Y_val_forecasting),
                                        callbacks=callbacks)
        else:
            log = self.model.fit(self.dataset.X_train, 
                                        self.dataset.X_train,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        validation_data=(self.dataset.X_val, self.dataset.X_val),
                                        callbacks=callbacks)
        self.log["loss"].extend(log.history["loss"])
        self.log["val_loss"].extend(log.history["val_loss"])
        
    def show_log(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.log["loss"])
        plt.plot(self.log["val_loss"])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        plt.close("all")
    
    def get_errors_vectors(self, y_true, y_hat, win_size):
        errors = []
        for i in range(win_size-1, len(y_hat)):
            v = [y_hat[i-s][s] for s in range(0, win_size)]
            v.reverse()
            pred_vector = np.array(v)
            error = np.abs((pred_vector - y_true[i]))
            errors.append(error)
        return np.array(errors)

    def get_validation_performance(self):
        if self.model_name in ["LSTM", "GRU"]:
            Y_hat = self.model.predict(self.dataset.X_val_forecasting)
            mae = np.mean(np.abs(Y_hat-self.dataset.Y_val_forecasting))
            mse = np.mean(np.abs(Y_hat-self.dataset.Y_val_forecasting)**2)
            print("MAE:", mae)
            print("MSE", mse)
        
        elif self.model_name in ["LSTM-multisteps", "GRU-multisteps"]:
            Y_hat = self.model.predict(self.dataset.X_val_forecasting)
            errors_vectors = self.get_errors_vectors(self.dataset.Y_val_forecasting, Y_hat, self.dataset.ahead_steps)
            print("MU:", np.mean(errors_vectors, axis=0))
            print("COV:", np.cov(errors_vectors.T))
        
        else:
            Y_hat = self.model.predict(self.dataset.X_val)
            mse = np.mean(np.abs(Y_hat-self.dataset.X_val)**2)
            print("MSE", mse)

    def _create_model(self):
        if self.model_name == "LSTM":
            inputs = Input(shape=(self.dataset.X_train_forecasting.shape[1], self.dataset.X_train_forecasting.shape[2]))
            z = LSTM(32, return_sequences=True)(inputs)
            z = LSTM(32)(z)
            outputs = Dense(self.dataset.Y_train_forecasting.shape[1])(z)

            model = Model(inputs, outputs, name=self.model_name)
            model.compile(loss="mae", optimizer=keras.optimizers.Adam(), metrics=["mae"])

        elif self.model_name == "GRU":
            inputs = Input(shape=(self.dataset.X_train_forecasting.shape[1], self.dataset.X_train_forecasting.shape[2]))
            z = GRU(32, return_sequences=True)(inputs)
            z = GRU(32)(z)
            outputs = Dense(self.dataset.Y_train_forecasting.shape[1])(z)

            model = Model(inputs, outputs, name=self.model_name)
            model.compile(loss="mae", optimizer=keras.optimizers.Adam(), metrics=["mae"])
        
        elif self.model_name == "LSTM-multisteps":
            inputs = Input(shape=(self.dataset.X_train_forecasting.shape[1], self.dataset.X_train_forecasting.shape[2]))
            z = LSTM(64, return_sequences=True)(inputs)
            z = LSTM(32)(z)
            outputs = Dense(self.dataset.Y_train_forecasting.shape[1])(z)

            model = Model(inputs, outputs, name=self.model_name)
            model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["mse"])
        
        elif self.model_name == "GRU-multisteps":
            inputs = Input(shape=(self.dataset.X_train_forecasting.shape[1], self.dataset.X_train_forecasting.shape[2]))
            z = GRU(64, return_sequences=True)(inputs)
            z = GRU(32)(z)
            outputs = Dense(self.dataset.Y_train_forecasting.shape[1])(z)

            model = Model(inputs, outputs, name=self.model_name)
            model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["mse"])

        elif self.model_name == "LSTM-Autoencoder":
            inputs = Input(shape=(self.dataset.X_train.shape[1], self.dataset.X_train.shape[2]))
            z = LSTM(128, return_sequences=True)(inputs)
            z = LSTM(64)(z)
            z = RepeatVector(self.dataset.X_train.shape[1])(z)
            z = LSTM(64, return_sequences=True)(z)
            z = LSTM(128, return_sequences=True)(z)
            outputs = TimeDistributed(Dense(self.dataset.X_train.shape[2]))(z)

            model = Model(inputs, outputs, name=self.model_name)
            model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["mse"])

        elif self.model_name == "GRU-Autoencoder":
            inputs = Input(shape=(self.dataset.X_train.shape[1], self.dataset.X_train.shape[2]))
            z = GRU(128, return_sequences=True)(inputs)
            z = GRU(64)(z)
            z = RepeatVector(self.dataset.X_train.shape[1])(z)
            z = GRU(64, return_sequences=True)(z)
            z = GRU(128, return_sequences=True)(z)
            outputs = TimeDistributed(Dense(self.dataset.X_train.shape[2]))(z)


            model = Model(inputs, outputs, name=self.model_name)
            model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=["mse"])
        
        model.summary()

        return model