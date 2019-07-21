import tensorflow as tf
from .base import Layer, Dense


class ProductAttention(Layer):
    def __init__(self, **kwargs):
        super(ProductAttention, self).__init__(**kwargs)

    def forward(self, query, value):
        pass


class ScaledProductAttention(Layer):
    def __init__(self, **kwargs):
        super(ScaledProductAttention, self).__init__(**kwargs)

    def forward(self, query, value):
        pass


class BiLinearAttention(Layer):
    def __init__(self, **kwargs):
        super(BiLinearAttention, self).__init__(**kwargs)

    def forward(self, query, value):
        pass


class MultiHeadAttention(Layer):
    def __init__(self, **kwargs):

        super(MultiHeadAttention, self).__init__(**kwargs)

    def forward(self, query, key, value):
        pass
