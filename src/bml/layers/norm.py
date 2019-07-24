from .base import Layer, Dense
import tensorflow as tf


class LayerNorm(Layer):
    def __init__(self, input_dims, axis=-1, gamma_initer='ones', beta_initer='zeros',
                 epsilon=1e-8, partitioner=None, name=None, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            self._gamma = tf.get_variable(
                name='gamma',
                shape=(input_dims,),
                partitioner=partitioner,
                dtype=tf.float32,
                initializer=gamma_initer,
                trainable=True
            )
            self._beta = tf.get_variable(
                name='beta',
                shape=(input_dims,),
                partitioner=partitioner,
                dtype=tf.float32,
                initializer=beta_initer,
                trainable=True
            )
        super(LayerNorm, self).__init__(**kwargs)

    # TODO: test
    def forward(self, inputs):
        mean, var = tf.nn.moments(inputs, self.axis)
        outputs = (inputs - mean) / (tf.sqrt(var) + self.epsilon)
        outputs = tf.multiply(outputs,self._gamma) + self._beta
        return outputs
