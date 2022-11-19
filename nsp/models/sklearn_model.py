import pickle
import time

from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE

from .model import Model


class LinearRegression(Model):
    def __init__(self):
        self.data = None
        self.model = None
        self.results = {}
        self.is_trained = False

    def train(self, data):
        self.data = data
        self.model = LR()

        _time = time.time()
        self.model.fit(self.data['x_tr'], self.data['y_tr'])
        _time = time.time() - _time

        self.results['time'] = _time
        self.is_trained = True

    def predict(self, x_in):
        assert self.is_trained
        return self.model(x_in)

    def eval_learning_metrics(self):
        assert self.is_trained

        self.results.update({
            'tr_mse': MSE(self.data['y_tr'], self.model.predict(self.data['x_tr'])),
            'tr_mae': MAE(self.data['y_tr'], self.model.predict(self.data['x_tr'])),
            'tr_mape': MAPE(self.data['y_tr'], self.model.predict(self.data['x_tr'])),
            'val_mse': MSE(self.data['y_val'], self.model.predict(self.data['x_val'])),
            'val_mae': MAE(self.data['y_val'], self.model.predict(self.data['x_val'])),
            'val_mape': MAPE(self.data['y_val'], self.model.predict(self.data['x_val'])),
        })

        print("* Linear Regression Results")
        print("** Train Scores:")
        print(f"      MSE:  {self.results['tr_mse']}")
        print(f"      MAE:  {self.results['tr_mae']}")
        print(f"      MAPE: {self.results['tr_mape']}")

        print("** Validation Scores:")
        print(f"      MSE:  {self.results['val_mse']}")
        print(f"      MAE:  {self.results['val_mae']}")
        print(f"      MAPE: {self.results['val_mape']}")

        print("  Linear regression train time:", self.results['time'], "\n")

    
    def save_model(self, fp):
        with open(fp, 'wb') as p:
            pickle.dump(self.model, p)

    def save_results(self, fp):
        with open(fp, 'wb') as p:
            pickle.dump(self.results, p)
