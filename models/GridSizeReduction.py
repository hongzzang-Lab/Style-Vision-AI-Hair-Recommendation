
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 
class BasicConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2D(out_channels, **kwargs)


    def call(self, x):
        return self.conv(x)
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 이 부분은 다른 파일로 끌어올꺼라 생각 x 

class Inception_ReduceA(tf.keras.Model):
    def __init__(self, in_channels):
        super(Inception_ReduceA, self).__init__()
        
        self.branchR1= tf.keras.Sequential([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding='same'),
            BasicConv2d(96, 96, kernel_size=3, strides=2)
        ])
        
        self.branchR2 = tf.keras.Sequential([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64,64,kernel_size=3, stride=2)])
        
        self.branchR3 = MaxPooling2D(pool_size=3, strides=2)

    def call(self, x):
        branchR1_out = self.branchR1(x)
        branchR2_out = self.branchR2(x)
        branchR3_out = self.branchR3(x)
        return Concatenate(axis=-1)([branchR1_out, branchR2_out, branchR3_out])

