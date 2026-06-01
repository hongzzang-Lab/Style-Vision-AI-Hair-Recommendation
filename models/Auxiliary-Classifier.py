import tensorflow as tf

from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Dropout, ReLU, Flatten

from tensorflow import layers

#BasicConv2d 는 나중에 모듈 다 합칠 떄 오류 해결 

class InceptionAux(tf.keras.Model):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avgpool = layers.AveragePooling2D(pool_size=(5,5), strides=3,padding = 'valid')
        self.conv = layers.Conv2D(128, kernel_size=1, strides=1, padding='valid',activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024,activation = 'relu')
        self.dropout = Dropout(0.7)
        self.fc2 = Dense(num_classes)

    def call(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x