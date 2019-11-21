import tensorflow as tf


class ScaledDotProductAtt(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(ScaledDotProductAtt, self).__init__(*args, **kwargs)

    def call(self, query, key, value, mask):
        """
        softmax(Q @ K^T/ sqrt(d_k)) @ V
        """
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_qk = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_qk += (mask * -1e9)
        att = tf.nn.softmax(scaled_qk, axis=-1)
        outputs = att @ value
        return outputs, att


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, *args, **kwargs):
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(
            units=d_model,
            use_bias=False
        )
        self.wk = tf.keras.layers.Dense(
            units=d_model,
            use_bias=False
        )
        self.wv = tf.keras.layers.Dense(
            units=d_model,
            use_bias=False
        )

        self.dense = tf.keras.layers.Dense(
            units=d_model,
            use_bias=False
        )

        self.scaled_dot_product_attention = ScaledDotProductAtt()

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]

        q = self.wq(query)  # (batch_size, seq_len_q, d_model)
        k = self.wk(key)  # (batch_size, seq_len_k, d_model)
        v = self.wv(value)  # (batch_size, seq_len_v, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights
