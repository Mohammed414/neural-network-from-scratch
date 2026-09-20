import numpy as np


# common Loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # clip the values of the y_pred for each element between (0 + 1e-7, 1 - 1e-7)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        
        # if the y_true is give a a 1D array
        if len(y_true.shape) == 1:
            correct_confidences = y_pred[
                range(samples),
                y_true
            ]

        # if the y_true is give as a 2x2 matrix 
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
