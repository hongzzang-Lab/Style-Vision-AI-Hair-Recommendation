import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Concatenate

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 

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


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 이 부분은 다른 파일로 끌어올꺼라 생각 x 



class InceptionB(tf.keras.Model):

    def __init__(self, input_channels,f_7x7):

        super(InceptionB, self).__init__()

        # branch 1: 1x1 conv &  double 1x7 and 7x1 convolutions

        self.branchB7x7stack = tf.keras.Sequential([
            BasicConv2d(input_channels,f_7x7, kernel_size=1),
            BasicConv2d(f_7x7,f_7x7, kernel_size=(1, 7), padding=(0,3)),
            BasicConv2d(f_7x7, f_7x7, kernel_size=(7, 1), padding=(3,0)),
            BasicConv2d(f_7x7, f_7x7, kernel_size=(1, 7), padding=(0,3)),
            BasicConv2d(f_7x7, 192, kernel_size=(7, 1), padding=(3,0))


        ])



        # branch 2: 1x1 conv & 1x7 followed by 7x1 (replacing large 7x7 convolutions)
        self.branchB7x7 = tf.keras.Sequential([
            BasicConv2d(input_channels, f_7x7, kernel_size=1),
            BasicConv2d(f_7x7,f_7x7,kernel_size=(1,7),padding = (0,3)) ,
            BasicConv2d(f_7x7, 192, kernel_size=(7, 1), padding= (3,0)) 
        ])

        # branch 3: Max Pooling + 1x1 convolution
        self.branchB_pool = tf.keras.Sequential([
            MaxPooling2D(pool_size=3, strides=1, padding='same'),
            BasicConv2d(input_channels, 192, kernel_size=1)
        ])

        # branch 4: 1x1 convolution
        self.branchB1x1 = BasicConv2d(input_channels, 192, kernel_size=1)

      
    def call(self, x):
        branchB2 = self.branchB7x7(x)
        branchB1 = self.branchB7x7stack(x)
        branchB4 = self.branchB1x1(x)
        branchB3 = self.branchB_pool(x)

        # Concatenate the outputs from the branches
        outputs = tf.concat([branchB1, branchB2, branchB3, branchB4], axis=-1)
        return outputs
    