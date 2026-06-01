import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D,MaxPooling2D, Concatenate


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
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

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 이 부분은 다른 파일로 끌어올꺼라 생각 x 

class InceptionC(tf.keras.Model):

    def __init__(self, input_channels):
        super(InceptionC, self).__init__()

        #branch 1
        self.branchC1= tf.keras.Sequential([
        BasicConv2d(input_channels, 448, kernel_size=1), 
        BasicConv2d(448, 384, kernel_size=3,padding =1)])
        
        self.branchC1_left = BasicConv2d(384,384,kernel_size=(1,3),padding=(0,1))
        self.branchC1_right = BasicConv2d(384,384,kernel_size=(3,1),padding=(1,0))
        
        #branch 2
        self.branchC2=BasicConv2d(input_channels,384, kernel_size=1)
        self.branchC2_left = BasicConv2d(384,384,kernel_size = (1,3),padding = (0,1))
        self.branchC2_right = BasicConv2d(384,384,kernel_size = (3,1), padding = (1,0))

        #branch 3 
        self.branchC1x1 = BasicConv2d(input_channels,320, kernel_size =1)

        #branch 4 
        self.branchC_pool = tf.keras.Sequential([
            MaxPooling2D(pool_size=3, strides=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1)
        ])

    def call(self, x):
        branchC1 = self.branchC1(x)
        branchC1_left= self.branchC1_left(branchC1)
        branchC1_right = self.branchC1_right(branchC1)
        
        branchC2 = self.branchC2(x) 
        branchC2_left = self.branchC2_left(branchC2)
        branchC2_right = self.branchC2_right(branchC2)

        branchC3 = self.branchC_pool(x)
        branchC3_1 = self.branchC1x1(branchC3)
        
        branchC4 = self.branchC1x1(x)

        outputs =[branchC1_left,branchC1_right,branchC2_left,branchC2_right,branchC3_1,branchC4]
        return Concatenate(axis=-1)(outputs)
