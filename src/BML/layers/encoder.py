import tensorflow as tf

from .attention import MultiHeadAttention
from .base import Layer, PositionFeedForward
from .norm import LayerNorm


class TransformerEncoder(Layer):
    def __init__(self, name=None, partitioner=None, **kwargs):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            self._pffn = PositionFeedForward(
                input_dims=1,
                hidden_dims=1,
                outputs_dims=1,
                kernel_initer='ones',
                bias_initer='zeros',
                name='ffn',
                partitioner=partitioner
            )

            self._att_layer = MultiHeadAttention(
                hidden_size=1024,
                query_dims=1,
                key_dims=1,
                value_dims=1,
                outputs_dims=1,
                heads=8,
                partitioner=partitioner,
                name='multiheads'
            )

            self._ln_1 = LayerNorm(
                input_dims=1,
                axis=-1,
                gamma_initer='ones',
                beta_initer='zeros',
                partitioner=partitioner,
                name='layernorm_1'
            )

            self._ln_2 = LayerNorm(
                input_dims=1,
                axis=-1,
                gamma_initer='ones',
                beta_initer='zeros',
                partitioner=partitioner,
                name='layernorm_2'
            )
        super(TransformerEncoder, self).__init__(**kwargs)

    def forward(self, key, query, value):
        _, outputs = self._att_layer(key, query, value)
        outputs += value
        outputs = self._ln_1(outputs) + outputs

        outputs = self._pffn(outputs) + outputs

        outputs = self._ln_2(outputs)
        return outputs
