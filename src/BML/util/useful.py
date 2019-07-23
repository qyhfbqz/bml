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


def get_tensor_shape(tensor):
    return [-1 if d is None else d for d in tensor.shape]


def combine_last_two_dim(tensor):
    tensor_shape = get_tensor_shape(tensor)
    output_shape = tensor_shape[:-2] + [tensor_shape[-2]*tensor_shape[-1]]
    return tf.reshape(tensor, output_shape)
