# # For abstract method support
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from collections import defaultdict

############################################################################################################


class Callable(ABC):
    """
    Callable Sub-classes:
     - CategoricalAccuracy (./metrics.py)   - TODO
     - OneHotEncoder       (./onehot.py)    - TODO
     - Diffable            (.)              - DONE
    """

    def __call__(self, *args, **kwargs) -> np.array:
        """Lets `self()` and `self.call()` be the same"""
        return Tensor(self.call(*args, **kwargs))

    @abstractmethod
    def call(self, *args, **kwargs) -> np.array:
        """Pass inputs through function. Can store inputs and outputs as instance variables"""
        pass


class Weighted(ABC):

    @abstractproperty
    def weights(): pass

    @property
    def trainable_weights(self):
        return [w for w in self.weights if w.trainable]

    @property
    def non_trainable_weights(self):
        return [w for w in self.weights if not w.trainable]

    @property
    def trainable(self):
        return len(self.trainable_weights) > 0

    @trainable.setter
    def trainable(self, value):
        for w in self.trainable_weights:
            w.trainable = value


class Diffable(Callable, Weighted):
    """
    Diffable Sub-classes:
     - Dense            (./layers.py)           - TODO
     - LeakyReLU, ReLU  (./activations.py)      - TODO
     - Softmax          (./activations.py)      - TODO
     - MeanSquaredError (./losses.py)           - TODO
    """

    """Stores whether the operation being used is inside a gradient tape scope"""
    gradient_tape = None  # All-instance-shared variable

    def __call__(self, *args, **kwargs) -> np.array:
        """
        If there is a gradient tape scope in effect, perform AND RECORD the operation.
        Otherwise... just perform the operation and don't let the gradient tape know.
        """

        # The call method keeps track of method inputs and outputs
        self.argnames = self.call.__code__.co_varnames[1:]
        named_args = {self.argnames[i]: args[i] for i in range(len(args))}
        self.input_dict = {**named_args, **kwargs}
        self.inputs = [self.input_dict[arg]
                       for arg in self.argnames if arg in self.input_dict.keys()]
        self.outputs = self.call(*args, **kwargs)

        # Make sure outputs are tensors and tie back to this layer
        list_outs = isinstance(self.outputs, list) or isinstance(
            self.outputs, tuple)
        if not list_outs:
            self.outputs = [self.outputs]

        # If Diffable has an active gradient_tape, go ahead and start recording backwards pathways
        if Diffable.gradient_tape is not None:
            for out in self.outputs:
                Diffable.gradient_tape.prevs[id(out)] = self

        # print(self.__class__.__name__.ljust(24), [v.shape for v in self.inputs], '->', [v.shape for v in self.outputs])

        # And then finally, it returns the output, thereby wrapping the forward call
        return self.outputs if list_outs else self.outputs[0]

    ################################################################################
    # Weights and weight-related properties
    @property
    def weights(self):
        """Returns weigths that could be trainable (or not)"""
        return []

    ################################################################################
    # Gradient-related methods
    def input_gradients(self):
        """Returns gradient for input (this part gets specified for all diffables)"""
        return []

    def weight_gradients(self):
        """Returns gradient for weights (this part gets specified for SOME diffables)"""
        return []

    def compose_to_input(self, J=None):
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied
        """
        if J is None or J[0] is None:
            return self.input_gradients()
        J_out = []
        for j in J:
            batch_size = j.shape[0]
            for v, g in zip(self.inputs, self.input_gradients()):
                j_new = np.zeros(v.shape, dtype=g.dtype)
                for b in range(batch_size):
                    g_b = g[b] if len(g.shape) == 3 else g
                    try:
                        j_new[b] = g_b @ j[b]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b @ j[b]` with {g_b.shape} and {j[b].shape}:\n{e}")
                J_out += [j_new]
        return J_out

    def compose_to_weight(self, J=None) -> list:
        """
        Compose the inputted cumulative jacobian with the weight jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `weight_gradients` to provide either batched or overall jacobian.
        Assumes weight/cumulative jacobians are element-wise multiplied (w/ broadcasting)
        and the resulting per-batch statistics are averaged together for avg per-param gradient.
        """
        if J is None or J[0] is None:
            return self.weight_gradients()
        J_out = []
        # For every weight/weight-gradient pair...
        for j in J:
            for v, g in zip(self.weights, self.weight_gradients()):
                batch_size = j.shape[0]
                # Make a cumulative jacobian which will contribute to the final jacobian
                j_new = np.zeros((batch_size, *v.shape), dtype=g.dtype)
                # For every element in the batch (for a single batch-level gradient updates)
                for b in range(batch_size):
                    # If the weight gradient is a batch of transform matrices, get the right entry.
                    # Allows gradient methods to give either batched or non-batched matrices
                    g_b = g[b] if len(g.shape) == 3 else g
                    # Update the batch's Jacobian update contribution
                    try:
                        j_new[b] = g_b * j[b]
                    except ValueError as e:
                        raise ValueError(
                            f"Error occured trying to perform `g_b * j[b]` with {g_b.shape} and {j[b].shape}:\n{e}")
                # The final jacobian for this weight is the average gradient update for the batch
                J_out += [np.sum(j_new, axis=0)]
            # After new jacobian is computed for each weight set, return the list of gradient updatates
        return J_out
    ################################################################################


############################################################################################################

class Tensor(np.ndarray):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.trainable = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.trainable = getattr(obj, 'trainable',  True)


Variable = Tensor

############################################################################################################


class GradientTape:

    def __init__(self):
        # Previous layers, as recorded by Diffable
        self.prevs = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, the diffables gradient_tape will be set so that the backwards
        # connetions can be properly recorded
        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, the diffables will no longer write backwards passes to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target, sources) -> list:
        # TODO:
        ##
        # Compute weight gradients for all operations.
        # If the model has trainable weights [w1, b1, w2, b2] and ends at a loss L.
        # the model should return: [dL/dw1, dL/db1, dL/dw2, dL/db2]
        ##
        # Breadth-first-search and record gradient histories for the Tensors.
        ##
        # Start from the last operation and compute jacobian w.r.t input.
        # Continue to propagate the cumulative jacobian through the layer inputs
        # until all operations have been differentiated through.
        ##
        # HINT: You will notice that id(ndarray) is used as the keys to dictionaries.
        # That's because ndarrays are not hashable.

        # TODO: Implement the algorithm of interest

        # 1. start at output of function - take first off of queue, use self.prevs, input grads, compose inputs, add to grads, use zip (for inputs, compose)
        # 2. work through generalized operations implemented in backward from hw1
        # 3. move backward following inputs and weights (BFS) -- use while loop? for loop? how to get self.prevs for each layer

        # how to get the inputs and weights from target (or any layer) -- i think target is a Diffable and if so we can use .inputs and .weights()
        # what is grads a dict of? layer to grad? one for weights and one for inputs?

        # Live queue; will be used to propagate backwards via breadth-first-search.
        queue = [target]
        # Grads to be recorded. Initialize to None
        grads = defaultdict(lambda: None)
        visited = [target]

        while queue:
            q = queue.pop(0)
            for i, g in zip(q.inputs, q.compose_to_input()): # q is Tensor??
                if Tensor.requires_grad:  # need? change?
                    if getattr(i, "requires_grad", False):
                        # i think i need something with id...
                        grads.update(i, g)
                    # need to backprop? i dont think we need backward() ?
            for (_, p), g in zip(enumerate(q.weights()), q.compose_to_weight()):
                if Tensor.requires_grad:
                    if getattr(p, "requires_grad", False):
                        grads.update(p, g)

            for n in self.prevs[q]:
                if n not in visited:
                    queue.append(n)
                    visited.append(n)

        # Retrieve the sources and make sure that all of the sources have been reached
        out_grads = [grads[id(source)][0] for source in sources]
        cut_graph = [f"var{i}" for i, grad in enumerate(
            out_grads) if grad is None]
        if cut_graph:
            print(
                f"Warning: The following tensors are disconnected from the target graph: {cut_graph}")

        return out_grads
