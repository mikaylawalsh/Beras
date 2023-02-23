from types import SimpleNamespace

import Beras
import numpy as np


class SequentialModel(Beras.Model):
    """
    Implemented in Beras/model.py

    def __init__(self, layers):
    def compile(self, optimizer, loss_fn, acc_fn):
    def fit(self, x, y, epochs, batch_size):
    def evaluate(self, x, y, batch_size):           ## <- TODO
    """

    def call(self, inputs):
        """Forward pass in sequential model. It's helpful to note that layers are initialized in Beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input)."""
        # TODO: What does it mean to call the model?
        # HINT: Beras_Intro...
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    def batch_step(self, x, y, training=True):
        """Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! Most of this method (, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()"""
        # TODO: Compute loss and accuracy for a batch. Return as a dictionary
        # If training, then also update the gradients according to the optimizer
        # HINT: Beras_Intro...

        # check training or not??

        with Beras.GradientTape() as tape:
            logits = self.call(x)
            loss = self.compiled_loss(logits, y)

            if training:
                grads = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(self.trainable_weights, grads)

        accuracy = self.compiled_acc(y, logits)

        # print loss and accuracy?
        return {"loss": loss, "acc": accuracy}


def get_simplest_model_components():
    """
    Returns a simple single-layer model. You can try running this one 
    as a first test if you'd like. This one will not be evaluated though.

    :return: model
    """

    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import BasicOptimizer

    model = SequentialModel([
        Dense(784, 10, initializer="normal"),
    ])
    model.compile(
        optimizer=BasicOptimizer(0.1),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=model, epochs=10, batch_size=256)


def get_simple_model_components():
    """
    Returns a simple model to satisfy 1470 portion.

    :return: model
    """

    # TODO: Create, compile, and return model.
    # Rquirements:
    # - Multiple layers
    # - LeakyRelu intermediate activation & sigmoid terminal activation
    # - Adam or RMSProp optimizer, MSE Loss, and Categorical Accuracy.

    # Default model provided

    from Beras.activations import LeakyReLU, Sigmoid
    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import Adam

    model = SequentialModel([
        Dense(784, 32, initializer="xavier uniform"),
        LeakyReLU(),
        Dense(32, 10, initializer="xavier uniform"),
        Sigmoid(),
    ])
    model.compile(
        optimizer=Adam(0.01),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=model, epochs=10, batch_size=256)


def get_advanced_model_components():
    """
    Returns a multi-layered model with more involved components.
    """

    # TODO: Implement a similar model, but make sure to use Softmax and CategoricalCrossentropy
    model = None
    return SimpleNamespace(model=model, epochs=10, batch_size=256)


if __name__ == "__main__":
    """
    Read in MNIST data, initialize your model, and train and test your model.
    """
    from Beras.onehot import OneHotEncoder
    from preprocess import load_and_preprocess_data

    # Read in MNIST data,
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()

    # Read in MNIST data, use the OneHotEncoder class to one hot encode the labels,
    # instantiate and compile your model, and train your model
    ohe = OneHotEncoder()
    concat_labels = np.concatenate([train_labels, test_labels], axis=-1)
    ohe.fit(concat_labels)

    # Threshold of accuracy:
    # 1470: >95% on testing accuracy from get_simple_model_components
    # 2470: >95% on testing accuracy from get_advanced_model_components + 1470 req
    arg_comps = [
        # get_simplest_model_components(),  # Simple starter option. Not graded
        get_simple_model_components(),  # 1470-required model; >95% accuracy
        # get_advanced_model_components()   ## 2470-required model; >95% accuracy
    ]
    for args in arg_comps:

        train_agg_metrics = args.model.fit(
            train_inputs,
            ohe(train_labels),
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        # Feel free to use the visualize_metrics function to view your accuracy and loss.
        # The final accuracy returned during evaluation must be > 80%.

        # from visualize import visualize_images, visualize_metrics
        # visualize_metrics(train_agg_metrics["loss"], train_agg_metrics["acc"])
        # visualize_images(model, train_inputs, ohe(train_labels))

        test_agg_metrics = args.model.evaluate(
            test_inputs, ohe(test_labels), batch_size=100)
        print("Testing Performance:", test_agg_metrics)
