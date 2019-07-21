import tensorflow as tf
from layers.base import Layer


class StackedDenoiseAutoEncoder(Layer):
    def __init__(self, **kwargs):
        super(StackedDenoiseAutoEncoder, self).__init__(**kwargs)

    def forward(self, inputs):
        pass
