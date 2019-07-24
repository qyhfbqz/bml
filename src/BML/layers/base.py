import tensorflow as tf
from abc import ABCMeta, abstractmethod
from ..util.useful import get_activation_func


class Layer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, *args):
        raise ValueError('Forward flow does not build!')

    def __call__(self, *args):
        return self.forward(*args)


class Dense(Layer):
    def __init__(self, input_dims, units, kernel_initer='ones',
                 bias_initer='zeros', use_bias=True, activation=None, partitioner=None,
                 name=None, **kwargs):
        self.activation_func = get_activation_func(activation)
        self.use_bias = use_bias

        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            self._kernel = tf.get_variable(
                name='kernel',
                shape=(input_dims, units),
                partitioner=partitioner,
                dtype=tf.float32,
                initializer=kernel_initer,
                trainable=True
            )
            if use_bias:
                self._bias = tf.get_variable(
                    name='bias',
                    shape=(1, units),
                    initializer=bias_initer,
                    dtype=tf.float32,
                    trainable=True,
                    partitioner=partitioner
                )

        super(Dense, self).__init__(**kwargs)

    def forward(self, inputs):
        outputs = tf.matmul(inputs, self._kernel)
        if self.use_bias:
            outputs += self._bias
        if self.activation_func:
            outputs = self.activation_func(outputs)
        return outputs


class Embedding(Layer):
    def __init__(self, num_ids, units, kernel_initer='ones',
                 name=None, partitioner=None, **kwargs):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            self._kernel = tf.get_variable(
                name='kernel',
                shape=(num_ids, units),
                partitioner=partitioner,
                dtype=tf.float32,
                initializer=kernel_initer,
                trainable=True
            )

        super(Embedding, self).__init__(**kwargs)

    def forward(self, inputs):
        outputs = tf.nn.embedding_lookup(
            params=self._kernel,
            ids=inputs,
            partition_strategy='div'
        )
        return outputs


class PositionFeedForward(Layer):
    def __init__(self, input_dims, hidden_dims, outputs_dims, kernel_initer='ones', bias_initer='zeros',
                 name=None, partitioner=None, **kwrgs):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            self._dense_1 = Dense(
                input_dims=input_dims,
                units=hidden_dims,
                kernel_initer=kernel_initer,
                bias_initer=bias_initer,
                name='dense_1',
                partitioner=partitioner,
                activation='relu'
            )

            self._dense_2 = Dense(
                input_dims=hidden_dims,
                units=outputs_dims,
                kernel_initer=kernel_initer,
                bias_initer=bias_initer,
                name='dense_2',
                partitioner=partitioner
            )

        super(PositionFeedForward, self).__init__(self)

    def forward(self, inputs):
        outputs = self._dense_1(inputs)
        outputs = self._dense_2(outputs)
        return outputs
