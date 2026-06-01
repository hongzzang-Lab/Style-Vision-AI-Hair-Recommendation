import tensorflow as tf
from module import BasicConv2d, Inception_ReduceA , InceptionA, InceptionB , InceptionC , InceptionAux ,Inception_ReduceB
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import layers, Model
from tensorflow.keras import layers

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

class InceptionV3(Model):
    def __init__(self,in_channels, num_classes, aux_logits= True, drop_p=0.5 , include_top = True, input_shape=(299,299)):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.include_top = include_top
       
       

        # Genrenal Conv + Polling
        
        self.conv1a = BasicConv2d(in_channels, 32, kernel_size=3, strides=2)
        self.conv1b = BasicConv2d(32, 32, kernel_size=3)
        self.conv1c = BasicConv2d(32, 64, kernel_size=3, padding=(1,1))

        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding = 'same')

        #pool_size =3 --> pooling 할 때 적용 커널의 크기가 3x3 이라는 뜻 
        # --> 299 x 299 size \
        self.conv2a = BasicConv2d(64, 80, kernel_size=3)
        self.conv2b = BasicConv2d(80, 192, kernel_size=3, strides=2)
        self.conv2c = BasicConv2d(192, 288, kernel_size=3, padding=(1,1))

        # --> 301 x 301 size 

        # Module A X 3 

        self.inception3a = InceptionA(input_channels= 288)
        self.inception3b = InceptionA(input_channels= 288)
        self.inception3c = InceptionA(input_channels= 288)

        # --> 301 x 301 size 

        # Grid Size Reduction 

        self.inception_red1 = Inception_ReduceA(288)

        # Module B X 5

        self.inception4a = InceptionB(768, f_7x7=128)
        self.inception4b = InceptionB(768, f_7x7=160)
        self.inception4c = InceptionB(768, f_7x7=160)
        self.inception4d = InceptionB(768, f_7x7=160)
        self.inception4e = InceptionB(768, f_7x7=192)


#ㅡㅡㅡㅡㅡㅡㅡㅡ 보조분류기 사용 유무 ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ# 

        if aux_logits:
            self.aux = InceptionAux(768, num_classes=num_classes)
        
        # Grid size Reduction 

        self.inception_red2 = Inception_ReduceB(768)

        self.conv2DT = Conv2D(192,(1,1))
        self.conv3DT = Conv2D(1280,(1,1))

        self.inception5a = InceptionC(1280)

        self.inception5b = InceptionC(2048)

        self.pool6 = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(rate=drop_p)
        self.fc = layers.Dense(num_classes)

        # Dens
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡForward ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

    def call(self, x, training=False):
        x = self.conv1a(x, training=training)
        x = self.conv1b(x, training=training)
        x = self.conv1c(x, training=training)

        x = self.pool1(x)

        x = self.conv2a(x, training=training)
        x = self.conv2b(x, training=training)
        x = self.conv2c(x, training=training)

        x = self.inception3a(x, training=training)
        x = self.inception3b(x, training=training)
        x = self.inception3c(x, training=training)

        x = self.inception_red1(x, training=training)

        x = self.inception4a(x, training=training)
        x = self.inception4b(x, training=training)
        x = self.inception4c(x, training=training)
        x = self.inception4d(x, training=training)
        x = self.inception4e(x, training=training)

        #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    
        aux = self.aux(x, training=training)


        x = self.inception5a(x, training=training)

        x = self.inception5b(x, training=training)
    
        x = self.pool6(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        print(x)
        #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
        
        return x ,aux  # x, aux를 결과값으로 출력한다.
    
 







