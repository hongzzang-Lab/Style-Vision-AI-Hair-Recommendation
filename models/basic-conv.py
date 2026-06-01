import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

class BasicConv2d(tf.keras.Model):

 # Conv2D Layer , batchnorm Layer , Relu함수를 정의
 
    def __init__(self, input_channels, output_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2D(output_channels, use_bias=False, **kwargs)
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
