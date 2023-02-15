import numpy as np

from .core import Diffable

################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Diffable):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def call(self, x):
        """Leaky ReLu forward propagation!"""
        pass ## TODO

    def input_gradients(self):
        """Leaky ReLu backpropagation!"""
        pass ## TODO


class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Diffable):
    
    ## TODO: Implement for default output activation to bind output to 0-1
    
    def call(self, x):
        pass ## TODO

    def input_gradients(self):
        pass ## TODO


class Softmax(Diffable):

    ## TODO [2470]: Implement for default output activation to bind output to 0-1

    def call(self, x):
        """Softmax forward propagation!"""
        ## HINT: Use stable softmax, which subtracts maximum from
        ## all entries to prevent overflow/underflow issues
        pass ## TODO

    def input_gradients(self):
        """Softmax input gradients!"""
        pass ## TODO

