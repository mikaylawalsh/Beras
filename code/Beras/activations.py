import numpy as np

from .core import Diffable

################################################################################
# Intermediate Activations To Put Between Layers


class LeakyReLU(Diffable):

    # TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def call(self, x):
        """Leaky ReLu forward propagation!"""
        # TODO
        # not sure if this is correct

        # ret = np.copy(x)
        # for i in range(len(x)):
        #     for j in range(len(x[i])):
        #         for k in range(len(x[i][j])):
        #             if x[i][j][k] > 0:
        #                 ret[i][j][k] = x[i][j][k]
        #             else:
        #                 ret[i][j][k] = self.alpha*x[i][j][k]
        
        ret = np.where(x > 0, x, x*self.alpha)
        return ret

    def input_gradients(self):
        """Leaky ReLu backpropagation!"""
        # TODO

        x = np.array(self.inputs)
        grad = np.copy(x)

        grad[x>0] = 1
        grad[x==0] = 0
        grad[x<0] = self.alpha

        # grad = np.copy(x)
        # for i in range(len(x)):
        #     for j in range(len(x[i])):
        #         for k in range(len(x[i][j])):
        #             for l in range(len(x[i][j][k])):
        #                 if x[i][j][k][l] > 0:
        #                     grad[i][j][k][l] = 1
        #                 elif x[i][j][k][l] == 0:
        #                     grad[i][j][k][l] = 0
        #                 else:
        #                     grad[i][j][k][l] = self.alpha
        return grad

    def compose_to_input(self, J):
        return self.input_gradients()[0] * J


class ReLU(LeakyReLU):
    # GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
# Output Activations For Probability-Space Outputs

class Sigmoid(Diffable):

    # TODO: Implement for default output activation to bind output to 0-1

    # need to fix to account for x as more than one value?

    def call(self, x):
        # TODO

        # ret = np.copy(x)
        # for i in range(len(x)):
        #     for j in range(len(x[i])):
        #         for k in range(len(x[i][j])):
                    
        # return ret

        ret = 1/(1+np.exp(-x))
        return ret



    def input_gradients(self):
        # TODO
        x = np.array(self.inputs)

        grad = np.copy(x)
        # for i in range(len(x)):
        #     for j in range(len(x[i])):
        #         for k in range(len(x[i][j])):
        #             for l in range(len(x[i][j][k])):
                        
        # return grad

        grad = np.exp(-x)/(1+np.exp(-x))**2
        return grad

    def compose_to_input(self, J):
        return self.input_gradients()[0] * J


class Softmax(Diffable):

    # TODO [2470]: Implement for default output activation to bind output to 0-1

    def call(self, x):
        """Softmax forward propagation!"""
        # HINT: Use stable softmax, which subtracts maximum from
        # all entries to prevent overflow/underflow issues
        pass  # TODO

    def input_gradients(self):
        """Softmax input gradients!"""
        pass  # TODO
