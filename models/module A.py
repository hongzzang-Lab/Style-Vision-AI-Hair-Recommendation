import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Concatenate

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 
class BasicConv2d(tf.keras.Model):

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
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#  이 부분은 다른 파일로 끌어올꺼라 생각 x 

class InceptionA(tf.keras.Model):

    def __init__(self, input_channels):
        super(InceptionA, self).__init__()

        # branch 1
        self.branchA3x3_double = tf.keras.Sequential([
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding='same'),
            BasicConv2d(96, 96, kernel_size=3, padding='same')
        ])

        # branch 2 
        self.branchA3x3 = tf.keras.Sequential([
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=3, padding='same')
        ])

        # branch 3
        self.branchApool = tf.keras.Sequential([
            layers.MaxPooling2D(pool_size=3, strides=1, padding='same'),
            BasicConv2d(input_channels,64, kernel_size=1)
        ])

        # branch 4
        self.branchA1x1 = BasicConv2d(input_channels, 64, kernel_size=1)



# 클래스의 순전파를 정의 forward 

    def call(self, x):
        branchA1 = self.branchA3x3_double(x)
        branchA2 = self.branchA3x3(x)
        branchA3 = self.branchApool(x)
        branchA4 = self.branchA1x1(x)

# 각 분기의 출력을 리스트로 묶음
        outputs = [branchA1,branchA2,branchA3,branchA4]