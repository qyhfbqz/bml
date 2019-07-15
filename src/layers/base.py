import tensorflow as tf
from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):
        pass



