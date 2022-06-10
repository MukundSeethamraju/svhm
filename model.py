import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, Dropout, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD

    
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, nb_filter, pool_size, alpha=0.1):
        super(ConvBlock, self).__init__()
        self.layer_1 = Conv2D(nb_filter, kernel_size=kernel_size)
        self.layer_2 = tf.keras.layers.LeakyReLU()
        self.layer_3 = Conv2D(nb_filter, kernel_size=kernel_size, padding='same')
        self.layer_4 = tf.keras.layers.LeakyReLU()
        self.layer_5 = MaxPooling2D(pool_size=pool_size)
        self.layer_6 = Dropout(alpha)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return x
        
        
class LargerConvBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, nb_filter, pool_size):
        super(LargerConvBlock, self).__init__()
        self.layer_1 = Conv2D(nb_filter, kernel_size=kernel_size)
        self.layer_2 = tf.keras.layers.LeakyReLU()
        self.layer_3 = Conv2D(nb_filter, kernel_size=kernel_size)
        self.layer_4 = tf.keras.layers.LeakyReLU()
        self.layer_5 = Conv2D(nb_filter, kernel_size=kernel_size, padding='same')
        self.layer_6 = tf.keras.layers.LeakyReLU()
        self.layer_7 = MaxPooling2D(pool_size=pool_size)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        return x


class PredictionBlock(tf.keras.layers.Layer):
    def __init__(self, nb_classes, activation):
        super(PredictionBlock, self).__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(1024, activation=tf.keras.layers.ReLU())
        self.layer_3 = Dropout(0.3) 
        # self.layer_4 = Dense(512, activation=tf.keras.layers.ReLU())
        # self.layer_5 = Dropout(0.3) 
        self.layer_6 = Dense(nb_classes, activation=activation)
        self.layer_7 = Dense(nb_classes, activation=activation)
        self.layer_8 = Dense(nb_classes, activation=activation)
        self.layer_9 = Dense(nb_classes, activation=activation)
        self.layer_10 = Dense(nb_classes, activation=activation)
    
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)
        # x = self.layer_5(x)
        x1 = self.layer_6(x)
        x2 = self.layer_7(x)
        x3 = self.layer_8(x)
        x4 = self.layer_9(x)
        x5 = self.layer_10(x)
        return [x1, x2, x3, x4, x5]


class PredictionBlock2(tf.keras.layers.Layer):
    def __init__(self, nb_classes, activation):
        super(PredictionBlock2, self).__init__()
        self.layer_1 = Flatten()
        self.layer_2 = Dense(256, activation=tf.keras.layers.ReLU())
        self.layer_3 = Dropout(0.5) 
        self.layer_4 = Dense(256, activation=tf.keras.layers.ReLU())

        self.layer_5 = Dense(nb_classes, activation=activation)
        self.layer_6 = Dense(nb_classes, activation=activation)
        self.layer_7 = Dense(nb_classes, activation=activation)
        self.layer_8 = Dense(nb_classes, activation=activation)
        self.layer_9 = Dense(nb_classes, activation=activation)
    
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x1 = self.layer_5(x)
        x2 = self.layer_6(x)
        x3 = self.layer_7(x)
        x4 = self.layer_8(x)
        x5 = self.layer_9(x)
        return [x1, x2, x3, x4, x5]


class SimpleModel(tf.keras.Model):
    def __init__(self, kernel_size, nb_filter, pool_size, activation='softmax', num_neurons=11):
        super(SimpleModel, self).__init__()
        self.block_1 = ConvBlock(kernel_size, 1, nb_filter, pool_size)
        self.block_2 = ConvBlock(kernel_size, 2, nb_filter, pool_size)
        # self.block_3 = ConvBlock(kernel_size, 4, nb_filter, pool_size) #! experimenting with smaller model
        self.prediction = PredictionBlock(num_neurons, activation)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        # x = self.block_3(x) #! experimenting with smaller model
        return self.prediction(x)



class VGGFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, trainable=False, weights='imagenet'):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = tf.keras.applications.vgg16.VGG16(include_top=False)
        self.vgg.trainable = trainable
    
    def call(self, inputs):
        x = self.vgg(inputs)
        return x


class VGGModel(tf.keras.Model):
    def __init__(self,  trainable=False, weights='imagenet', activation='softmax', num_classes=11):
        super(VGGModel, self).__init__()
        self.vgg = VGGFeatureExtractor()
        self.prediction = PredictionBlock(num_classes, activation)

    def call(self, inputs):
        x = self.vgg(inputs)
        return self.prediction(x)



class SimpleModel2(tf.keras.Model):
    def __init__(self, kernel_size, nb_filter, pool_size, activation='softmax', num_neurons=11):
        super(SimpleModel2, self).__init__()
        self.layer_1 = Dropout(0.1)
        self.block_1 = ConvBlock(kernel_size, nb_filter, pool_size, alpha=0.25)
        self.block_2 = ConvBlock(kernel_size, 2*nb_filter, pool_size, alpha=0.25)
        self.block_3 = LargerConvBlock(kernel_size, 4*nb_filter, pool_size) 
        self.prediction = PredictionBlock2(num_neurons, activation)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x) 
        return self.prediction(x)

