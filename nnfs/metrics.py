# Includes functions to calculate training metrics like accuracy.

import numpy as np

class Metric:
    def __init__(self):
       self.epoch_stat = 0 # cumulative sum of the stat (e.g. cumulative epoch loss)
       self.name = ''
       self.r4 = lambda x: round(x, 4)

    def reset(self):
       self.epoch_stat = 0

    def update(self, y_pred, batch_y):
        pass

    def disp(self, batch_idx):
        return ' | {}: {}'.format(self.name, self.r4(self.epoch_stat / batch_idx))

class LossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.name = 'Loss'

    def update(self, loss):
        self.epoch_stat += loss

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.name = 'Accuracy'

    def update(self, y_pred, batch_y):
        y_correct = (np.argmax(y_pred, axis=1) == np.argmax(batch_y, axis=1)).astype(np.int32)
        accuracy = np.mean(y_correct)
        self.epoch_stat += accuracy
