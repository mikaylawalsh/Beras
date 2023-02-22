import numpy as np

from .core import Diffable, Variable


class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer="kaiming"):
        self.w, self.b = self.__class__._initialize_weight(
            initializer, input_size, output_size)

    @property
    def weights(self):
        return self.w, self.b

    def call(self, x):
        """Forward pass for a dense layer! Refer to lecture slides for how this is computed."""
        return np.matmul(x, self.w) + self.b

    def input_gradients(self):
        return [self.w]

    def weight_gradients(self):
        """Calculating the gradients of the weights and biases!"""
        x, y = self.inputs + self.outputs
        wgrads = np.ones_like(self.w) * np.expand_dims(x, axis=-1)
        bgrads = np.ones_like(self.b)
        return [wgrads, bgrads]

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size):
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
            "xavier uniform",
            "kaiming uniform",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"
        io_size = (input_size, output_size)

        w_init = np.zeros(io_size)
        b_init = np.zeros((1, output_size))

        if initializer == "normal":
            w_init = np.random.normal(size=io_size)

        # TODO: Implement remaining options (normal, xavier, kaiming initializations) for w_init.
        # Note that strings must be exactly as written in the assert above
        if initializer == "xavier":
            stddev = (2 / (input_size + output_size))**(.5)
            w_init = np.random.normal(0, stddev, io_size)

        if initializer == "kaiming":
            stddev = (2 / input_size)**(.5)
            w_init = np.random.normal(0, stddev, io_size)

        if initializer == "xavier uniform":
            limit = (6 / (input_size + output_size))**(.5)
            w_init = np.random.uniform(-limit, limit, io_size)

        if initializer == "kaiming uniform":
            limit = (6 / input_size)**(.5)
            w_init = np.random.uniform(-limit, limit, io_size)

        return Variable(w_init), Variable(b_init)
