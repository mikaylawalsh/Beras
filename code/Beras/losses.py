import numpy as np

from .core import Diffable

################################################################################
## 1470 Default Loss Function

class MeanSquaredError(Diffable):

    ## HINT: If only Beras had an intro notebook that talked about it...

    def __init__(self, *args, from_logits=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def call(self, y_pred, y_true):
        ## If not from_logits (i.e. from softmax), only consider prediction on true class
        """Mean squared error forward pass!"""
        if self.from_logits:
            mse_total = np.mean(np.power(y_pred - y_true, 2), axis=-1)
        else: 
            mse_total = np.mean(np.power(y_pred * y_true - y_true, 2), axis=-1)
        return np.mean(mse_total, axis=0)

    def input_gradients(self):
        """Mean squared error input gradient for both input arguments!"""
        ## Remember to normalize!
        y_pred, y_true = self.inputs
        if self.from_logits:
            ## Using all entries. Sub-par if using softmax
            grad = 2 * (y_pred - y_true) / np.prod(y_pred.shape)
        else:
            ## Only using target class. Better for softmax
            grad = 2 * ((y_pred * y_true) - y_true) / y_pred.shape[0]
        return [grad, -grad]


################################################################################
## 2470 Required Loss Function

def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1-eps)

class CategoricalCrossentropy(Diffable):

    ## TODO [2470]: Implement Stable CategoricalCrossentropy Function

    def call(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        pass ## TODO

    def input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        pass ## TODO
