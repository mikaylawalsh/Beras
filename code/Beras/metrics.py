from re import I
import numpy as np

from .core import Callable


class CategoricalAccuracy(Callable):
    def call(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '==' 
        
        accurate = 0
        total = 0
        # where/how to use boolean mask? 
        for i in range(len(probs)):
            if np.argmax(probs[i]) == np.argmax(labels[i]):
                accurate += 1
            total += 1 # better way to do this?
        
        return accurate/total