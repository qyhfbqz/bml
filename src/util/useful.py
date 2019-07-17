import tensorflow as tf

all_activation_func_map = {
    'linear': None,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}


def get_activation_func(activation):
    if isinstance(activation, str):
        return all_activation_func_map.get(activation.lower())
    else:
        return None
