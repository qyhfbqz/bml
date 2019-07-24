import tensorflow as tf

from ..util.useful import combine_last_two_dim, get_tensor_shape
from .base import Dense, Layer


class ProductAttention(Layer):
    # TODO deal inputs with different rank
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super(ProductAttention, self).__init__(**kwargs)

    def forward(self, query, key, value):
        return self.get_attention(query, key)

    def get_attention(self, query, key):
        outputs = tf.matmul(query, key)
        outputs = outputs / self.scale
        outputs = tf.nn.softmax(outputs)
        return outputs


class BiLinearAttention(Layer):
    def __init__(self, **kwargs):
        super(BiLinearAttention, self).__init__(**kwargs)

    def forward(self, query, key, value):
        pass

    def get_attention(self, query, key):
        pass


class MultiHeadAttention(Layer):
    # TODO add mask inputs
    def __init__(self, hidden_size, query_dims, key_dims, value_dims, heads,
                 outputs_dims, project_kernel_initer='ones', use_bias=False,
                 activation='relu', return_attention=False, name=None,
                 partitioner=None, **kwargs):
        self._d_k = hidden_size // heads
        if heads * self._d_k != hidden_size:
            raise ValueError('d_k')
        self._scale = self._d_k ** 0.5
        self._heads = heads
        self._return_att = return_attention
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            self._query_project = Dense(
                input_dims=query_dims,
                units=hidden_size,
                kernel_initer=project_kernel_initer,
                name='query_project',
                activation=activation,
                use_bias=use_bias,
                partitioner=partitioner
            )

            self._key_project = Dense(
                input_dims=key_dims,
                units=hidden_size,
                kernel_initer=project_kernel_initer,
                name='key_project',
                activation=activation,
                use_bias=use_bias,
                partitioner=partitioner
            )

            self._value_project = Dense(
                input_dims=value_dims,
                units=hidden_size,
                kernel_initer=project_kernel_initer,
                name='value_project',
                activation=activation,
                use_bias=use_bias,
                partitioner=partitioner
            )

            self._att_func = ProductAttention(
                scale=self._scale
            )

            # TODO: add output_dims
            output_dims = None
            self._output_project = Dense(
                input_dims=output_dims,
                units=outputs_dims,
                kernel_initer=project_kernel_initer,
                name='output_project',
                activation=activation,
                use_bias=use_bias,
                partitioner=partitioner
            )

        super(MultiHeadAttention, self).__init__(**kwargs)

    def forward(self, query, key, value):
        query = self._query_project(query)
        query_shape = self.get_multi_head_shape(query)
        query = tf.reshape(query, query_shape)
        query = tf.transpose(query, [0, 2, 1, 3])

        key = self._key_project(key)
        key_shape = self.get_multi_head_shape(key)
        key = tf.reshape(key, key_shape)
        key = tf.transpose(key, [0, 2, 3, 1])

        attention = self.get_attention(query, key)

        value = self._value_project(value)
        value_shape = self.get_multi_head_shape(value)
        value = tf.reshape(value, value_shape)
        value = tf.transpose(value, [0, 2, 1, 3])

        outputs = tf.matmul(attention, value)
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        outputs = combine_last_two_dim(outputs)

        outputs = self._output_project(outputs)

        if self._return_att:
            return attention, outputs
        else:
            return outputs

    def get_attention(self, query, key):
        return self._att_func(query, key, None)

    def get_multi_head_shape(self, inputs):
        output_shape = get_tensor_shape(inputs)
        output_shape = [d for d in output_shape[:-1]]
        output_shape.extend([self._heads, self._d_k])
        return output_shape
