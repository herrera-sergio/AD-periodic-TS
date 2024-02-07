import pandas as pd

class EvaluateModel:

    def __init__(self,
                model,
                test_path):
        
        self.model = model
        self.X_test, self.X_test_forecasting, self.Y_test_forecasting, self.data = self.model.dataset.get_test_data(test_path)

    def generate_predictions(self, output_path):
        if self.model.model_name in ["LSTM-Autoencoder", "GRU-Autoencoder"]:
            Y_hat = self.model.model.predict(self.X_test)
        else:
            Y_hat = self.model.model.predict(self.X_test_forecasting)
        
        if self.model.model_name in ["LSTM", "GRU"]:
            output_data = pd.DataFrame({self.model.dataset.feature_name: Y_hat.reshape(-1)},
                                        index=self.data.index.values[self.model.dataset.sliding_window_size:])
        else:
            if self.model.model_name in ["LSTM-multisteps", "GRU-multisteps"]:
                win = self.model.dataset.ahead_steps
                index_values = self.data.index.values[self.model.dataset.sliding_window_size:]
            else:
                win = self.model.dataset.sliding_window_size
                index_values = self.data.index.values
            out_values = []
            Y_hat = Y_hat.reshape(-1, win)

            for i in range(0, len(Y_hat)):
                v = []
                if i < win-1:
                    for j in range(0, i+1):
                        v = [Y_hat[i-j][j]] + v
                else:
                    for j in range(0, win):
                        v = [Y_hat[i-j][j]] + v
                out_values.append(v)
                
            for i in range(len(Y_hat)-win+1, len(Y_hat)):
                v = []
                for k, j in enumerate(range(win-1, win-(len(Y_hat)-i)-1, -1)):
                    v = v + [Y_hat[i+k][j]]
                out_values.append(v)
            output_data = pd.DataFrame({self.model.dataset.feature_name: out_values},
                                        index=index_values)
        
        output_data.to_csv(output_path, index_label="ctime")