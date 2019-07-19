import tensorflow as tf
from .base import Layer, Dense


class MultiHeadAttention(Layer):
    def __init__(self, **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)

    def forward(self):
        pass
