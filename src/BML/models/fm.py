import tensorflow as tf
from ..layers.base import Layer


class FactorizationMachine(Layer):
    def __init__(self, input_dims, units, kernel_initer, use_bias=True,
                 partitioner=None, name=None, trainable=True, **kwargs):
        self.partitioner = partitioner
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(name_or_scope='one_order', reuse=tf.AUTO_REUSE):
                self._one_order_kernel = tf.get_variable(
                    name='kernel',
                    shape=(input_dims, 1),
                    partitioner=partitioner,
                    dtype=tf.float32,
                    initializer=kernel_initer,
                    trainable=trainable
                )
                if use_bias:
                    self._one_order_bias = tf.get_variable(
                        name='kernel',
                        shape=(1, 1),
                        partitioner=partitioner,
                        dtype=tf.float32,
                        initializer=kernel_initer,
                        trainable=trainable
                    )
                else:
                    self._one_order_bias = None

            with tf.variable_scope(name_or_scope='second_order', reuse=tf.AUTO_REUSE):
                self._second_order_kernel = tf.get_variable(
                    name='kernel',
                    shape=(input_dims, units),
                    partitioner=partitioner,
                    dtype=tf.float32,
                    initializer=kernel_initer,
                    trainable=trainable
                )

    def forward(self, inputs):
        one_order_outputs = tf.nn.embedding_lookup(
            params=self._one_order_kernel,
            ids=inputs,
            partitioner=self.partitioner,
            partition_strategy='div'
        )

        hidden_vector = tf.nn.embedding_lookup(
            params=self._second_order_kernel,
            ids=inputs,
            partitioner=self.partitioner,
            partition_strategy='div'
        )

        sum_then_square = tf.square(tf.reduce_sum(hidden_vector, 1))
        square_then_sum = tf.reduce_sum(tf.square(hidden_vector), 1)
        second_order_outputs = 0.5 * \
            tf.reduce_sum(sum_then_square-square_then_sum)

        outputs = one_order_outputs + second_order_outputs

        if self._one_order_bias:
            outputs += self._one_order_bias

        return outputs
