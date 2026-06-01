
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,ZeroPadding2D, BatchNormalization, ReLU, AveragePooling2D, Dense, Dropout,Flatten, MaxPooling2D, Concatenate
import sys
from tensorflow.keras import layers

sys.path.append('C:/Users/USER/AppData/Local/Programs/Python/Python312/Lib/site-packages')


# conv2D Layer는 padding 값으로 valid or same 만을 허용 
# padding = 'same'  : 출력 이미지의 크기 = 입력 이미지의 크기 
# Conv block 정의
# **kwargs : 나중에 이 class를 사용해서 인스턴스화 할 때 정의하지 않은 인수들도 전달 가능 

class BasicConv2d(tf.keras.Model):
    def __init__(self,in_channels, output_channels,kernel_size , padding = None ,**kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2D(output_channels,kernel_size = kernel_size ,**kwargs)
        self.padding = ZeroPadding2D(padding) if padding else None # padding 값이 주어지면 zeropadding layer 생성, 아니면 패딩 적용 x 
        self.bn = BatchNormalization(axis = -1 , momentum = 0.99, epsilon = 0.001, center = True , scale = True) 
        self.relu = ReLU()



    def call(self, x):
        if self.padding: 
            x = self.padding(x) 
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

class InceptionA(tf.keras.Model):

    def __init__(self, input_channels):
        super(InceptionA, self).__init__()

        # branch 1
        self.branchA3x3_double = tf.keras.Sequential([
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=(1,1)),
            BasicConv2d(96, 96, kernel_size=3, padding=(1,1))
        ]) 
        
        # branch 2 
        self.branchA3x3 = tf.keras.Sequential([
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=3, padding=(1,1))
        ])
        # branch 3 # MaxPooling 2D 인수 : Padding 은 'same' & 'valid' 허용 
        self.branchApool = tf.keras.Sequential([
            MaxPooling2D(pool_size=3, strides=1, padding='same'),
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

# Concatenate를 사용하여 각 분기의 출력을 채널 축 ( AXIS = -1 ) 으로 병함 --> 입력값이 4차원 tensor여야 함 
        return Concatenate(axis=-1)(outputs)
# ex) 마지막 채널수가 각각 a,b,c 라고 할 때 concatenate를 통하면 다 합한 (a+b+c)가 채널수로 나온다 

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#


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
            BasicConv2d(f_7x7,f_7x7,kernel_size=(1,7),padding = (0,3)),
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
    
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

class InceptionC(tf.keras.Model):

    def __init__(self, input_channels):
        super(InceptionC, self).__init__()

        #branch 1
        self.branchC1= tf.keras.Sequential([
        BasicConv2d(input_channels, 448, kernel_size=1), 
        BasicConv2d(448, 384, kernel_size=3,padding =1)])
        

        self.branchC1_left = BasicConv2d(384,384,kernel_size=(1,3),padding=((0,0), (1,1)))
        self.branchC1_right = BasicConv2d(384,384,kernel_size=(3,1),padding=((1,1), (0,0)))
        
        #branch 2
        self.branchC2=BasicConv2d(input_channels,384, kernel_size=1)
        self.branchC2_left = BasicConv2d(384,384,kernel_size = (1,3),padding =((0,0), (1,1)))
        self.branchC2_right = BasicConv2d(384,384,kernel_size = (3,1), padding = ((1,1), (0,0)))

        #branch 3 
        self.branchC_pool = tf.keras.Sequential([
            MaxPooling2D(pool_size=3, strides=1, padding='same'),
            BasicConv2d(input_channels,192 , kernel_size=1)
        ])

        #branch 4
        self.branchC1x1 = BasicConv2d(input_channels,320, kernel_size =1)

    def call(self, x):
        branchC1 = self.branchC1(x)
        branchC1_left= self.branchC1_left(branchC1)
        branchC1_right = self.branchC1_right(branchC1)
        
        branchC2 = self.branchC2(x) 
        branchC2_left = self.branchC2_left(branchC2)
        branchC2_right = self.branchC2_right(branchC2)

        branchC3 = self.branchC_pool(x)
        #branchC3_1 = self.branchC1x1(branchC3)
        
        branchC4 = self.branchC1x1(x)
        #print(branchC1_left.shape ,branchC1_right.shape,branchC2_left.shape,branchC2_right.shape,branchC3.shape,branchC4.shape)
        outputs =[branchC1_left,branchC1_right,branchC2_left,branchC2_right,branchC3,branchC4]
        return Concatenate(axis=-1)(outputs)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
# padding = 'valid' 는 zeropadding 에서 padding 인수를 생략하는 것과
class InceptionAux(tf.keras.Model):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = layers.AveragePooling2D(pool_size=(5,5), strides=3)
        self.conv = layers.Conv2D(filters = 128, kernel_size=1, strides=1,activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024,activation = 'relu')
        self.dropout = layers.Dropout(0.7)
        self.fc2 = layers.Dense(num_classes)

#Dense ( 5 ) --> 입력 data 를 5개의 출력 노드로 변환한다. 

    def call(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#


class Inception_ReduceA(tf.keras.Model):
    def __init__(self, in_channels):
        
        super(Inception_ReduceA, self).__init__()

        self.branchR1= tf.keras.Sequential([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=(1,1)),
            BasicConv2d(96, 96, kernel_size=3, strides=2)
        ])

        self.branchR2 = BasicConv2d(in_channels , 384 , kernel_size=3 , strides = 2)

        self.branchR3 = MaxPooling2D(pool_size=3, strides=2)


    def call(self, x):
        
        branchR1_out = self.branchR1(x)
        branchR2_out = self.branchR2(x)
        branchR3_out = self.branchR3(x)

        return Concatenate(axis=-1)([branchR1_out, branchR2_out, branchR3_out])
    
    
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

class Inception_ReduceB(tf.keras.Model):
    def __init__(self, in_channels):
        super(Inception_ReduceB, self).__init__()

        self.branch1 = tf.keras.Sequential([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=((0, 0 ), (3,3))),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=((3,3),(0,0))),
            BasicConv2d(192, 192, kernel_size=3, strides=2)
        ])

        self.branch2 = tf.keras.Sequential([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, strides=2)
        ])

        self.branch3 = MaxPooling2D(pool_size=3, strides=2)

    def call(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        return Concatenate(axis=-1)([branch1, branch2, branch3])

