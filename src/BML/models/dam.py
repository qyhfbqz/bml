import tensorflow as tf
from ..layers.base import Layer


class DeepAttentionMatch(Layer):
    """An implementation of Multi-Turn Response Selection for Chatbots with Deep Attention
    Matching Network(https://aclweb.org/anthology/P18-1103)

    """

    def __init__(self, **kwargs):
        super(DeepAttentionMatch, self).__init__(**kwargs)

    def forward(self, query, context):
        """Forward.
            
        Args:
            query: query sequences, shape=(B,N)
            context: context sequences, shape=(B,C,N)

        Returns:
            

        Raises:
            

        """
        pass
