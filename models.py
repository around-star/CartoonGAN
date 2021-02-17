import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LayerNormalization, ReLU, LeakyReLU, Conv2DTranspose, ZeroPadding2D, UpSampling2D
from tensorflow.keras import Input, Model

# Generator Model
class ResBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.block = Sequential(layers = [Conv2D(filters = 256, kernel_size = (3,3), padding='same'),
                                          LayerNormalization(),
                                          ReLU(),
                                          Conv2D(filters = 256, kernel_size = (3,3), padding='same'),
                                          LayerNormalization()
                                          ])
        self.activation = ReLU()

    def call(self, input):
        output = self.block(input)
        output += input
        output = self.activation(output)

        return output

def generator():
    input = Input(shape = (None, None, 3))

    # flat convolution block
    conv1_1 = Conv2D(filters = 64, kernel_size = (7,7), padding = 'same')(input)
    norm1_1 = LayerNormalization()(conv1_1)
    activation1_1 = ReLU()(norm1_1)

    # down convolution blocks
    pad2_1 = ZeroPadding2D()(activation1_1)
    conv2_1 = Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2))(pad2_1)
    conv2_2 = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same')(conv2_1)
    norm2_1 = LayerNormalization()(conv2_2)
    activation2_1 = ReLU()(norm2_1)

    pad3_1 = ZeroPadding2D()(activation2_1)
    conv3_1 = Conv2D(filters = 256, kernel_size = (3,3), strides = (2,2))(pad3_1)
    conv3_2 = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same')(conv3_1)
    norm3_1 = LayerNormalization()(conv3_2)
    activation3_1 = ReLU()(norm3_1)

    # residual blocks
    residual_block_1 = ResBlock()(activation3_1)
    residual_block_2 = ResBlock()(residual_block_1)
    residual_block_3 = ResBlock()(residual_block_2)
    residual_block_4 = ResBlock()(residual_block_3)
    residual_block_5 = ResBlock()(residual_block_4)
    residual_block_6 = ResBlock()(residual_block_5)
    residual_block_7 = ResBlock()(residual_block_6)
    residual_block_8 = ResBlock()(residual_block_7)

    # up convolution blocks
    conv4_1 = Conv2DTranspose(filters = 128, kernel_size = (2,2), strides = (2, 2))(residual_block_8)
    #conv4_2 = Conv2DTranspose(filters = 128, kernel_size = (3,3))(pad4_2)
    norm4_1 = LayerNormalization()(conv4_1)
    activation4_1 = ReLU()(norm4_1)

    conv5_1 = Conv2DTranspose(filters = 64, kernel_size = (2,2), strides = (2, 2))(activation4_1)
    #conv5_2 = Conv2DTranspose(filters = 64, kernel_size = (3,3))(conv5_1)
    norm5_1 = LayerNormalization()(conv5_1)
    activation5_1 = ReLU()(norm5_1)

    # output
    output = Conv2D(filters = 3, kernel_size = (7,7), padding='same')(activation5_1)

    model = Model(inputs = input, outputs = output)

    return model



#Discriminator Model

class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, pad = (1, 1)):
        super(ReflectionPad2D, self).__init__()
        padding = tuple (pad)
        self.padding = ((0,0), padding, padding, (0, 0))

    def call(self, x):
        return tf.pad(x, self.padding, "REFLECT")

class StridedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size = (1,1), strides = (1,1)):
        super(StridedConv2D, self).__init__()
        self.block = Sequential(layers = [ReflectionPad2D(),
                                          Conv2D(filters, kernel_size, strides),
                                          LeakyReLU(alpha = 0.2),
                                          ReflectionPad2D(),
                                          Conv2D(filters * 2, kernel_size)])
        
    def call(self, input):
        
        output = self.block(input)
        return output

def discriminator():
    input = Input(shape = (None, None, 3))

    conv1_1 = Conv2D(filters = 32, kernel_size = (3,3))(input)
    activation1_1 = LeakyReLU(alpha = 0.2)(conv1_1)

    conv2_1 = StridedConv2D(filters = 64, strides = (2,2))(activation1_1)
    conv3_1 = StridedConv2D(filters = 128, strides = (2,2))(conv2_1)

    conv4_1 = Conv2D(filters = 256, kernel_size = (3,3))(conv3_1)
    norm4_1 = LayerNormalization()(conv4_1)
    activation4_2 = LeakyReLU(alpha = 0.2)(norm4_1)

    output = Conv2D(filters = 1, kernel_size = (3,3))(activation4_2)

    model = Model(inputs = input, outputs = output)

    return model