# custom L1Distance Layer

#Import dependancies
import tensorflow as tf
from tensorflow.keras.layers import Layer 

# custom L1Dist from the model 
# L1 distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)