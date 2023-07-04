from matplotlib import pyplot as plt
from statsmodels import api as sm
from typing import List
import pandas as pd
import numpy as np


class LogisticRegression:
    def __init__(self, data: pd.core.frame.DataFrame, inputs: List[str], output: str):
        self.data = data
        self.inputs = inputs
        self.output = output
        self.log_reg = None

    def _data_split(self):
        return sm.add_constant(self.data[self.inputs]), self.data[self.output]

    def _create_regression(self, x, y):
        reg = sm.Logit(y, x)
        self.log_reg = reg.fit()

    def _summary_table(self):
        return self.log_reg.summary()

    def _confusion_matrix(self):
        matrix = pd.DataFrame(self.log_reg.pred_table())
        matrix = matrix.rename(index={
            0: 'Actual 0',
            1: 'Actual 1'
        })
        matrix.columns = ['Predicted 0', 'Predicted 1']

        return matrix

    def _calculate_fitting_percentage(self):
        cm = np.array(self.log_reg.pred_table())
        at = (cm[0, 0] + cm[1, 1]) / cm.sum()
        return (at * 100).round(3)

    def main(self):
        result = {}
        x, y = self._data_split()
        self._create_regression(x, y)

        result['Summary'] = self._summary_table()
        result['Matrix'] = self._confusion_matrix()
        result['Percentage'] = self._calculate_fitting_percentage()

        return result
