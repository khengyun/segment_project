import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
smooth = 1.
dropout_rate = 0.5


# Custom loss function
# def dice_coef(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(K.dot(y_true_f , y_pred_f))
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_loss(y_true, y_pred):
#     return 1 - dice_coef(y_true, y_pred)

# def bce_dice_loss(y_true, y_pred):
#     return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Input
# from tensorflow.python.keras.engine import training
from tensorflow.python.keras.models import Model

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices) )

class UNetPlusPlus():
    
    def __init__(self, input_shape = (128, 128, 3), filters = [16,32, 64, 128, 256], nb_classes = 1, deep_supervision = False):
        
       
        self.input__shape = input_shape
        self.filters = filters
        self.num_classes = nb_classes
        self.deep_supervision = deep_supervision
        self.__smooth = 1. 
    
    def BuildNetwork(self):


        input_img = Input(shape = (128, 128, 3), name = 'InputLayer')
        
        conv00 = self.AadyasConvolutionBlock(input_img, block_level = '00', filters = self.filters[0])
        pool0 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool0')(conv00)

        conv10 = self.AadyasConvolutionBlock(pool0, block_level = '10', filters = self.filters[1])
        pool1 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool1')(conv10)

        up01 = Conv2DTranspose(filters = self.filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample01')(conv10)
        conv01 = concatenate([up01, conv00], name='concat01')
        conv01 = self.AadyasConvolutionBlock(conv01, block_level = '01', filters = self.filters[0])

        conv20 = self.AadyasConvolutionBlock(pool1, block_level = '20', filters = self.filters[2])
        pool2 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool2')(conv20)

        up11 = Conv2DTranspose(filters = self.filters[1], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample11')(conv20)
        conv11 = concatenate([up11, conv10], name = 'concat11')
        conv11 = self.skip_path(conv11, block_level = '11', filters = self.filters[1])

        up02 = Conv2DTranspose(filters = self.filters[0], kernel_size = 2, strides = 2, padding='same', name='upsample02')(conv11)
        conv02 = concatenate([up02, conv00, conv01], name = 'concat02')
        conv02 = self.AadyasConvolutionBlock(conv02, block_level = '02', filters = self.filters[0])

        conv30 = self.AadyasConvolutionBlock(pool2, block_level = '30', filters = self.filters[3])
        pool3 = MaxPooling2D(pool_size = 2, strides = 2, name = 'pool3')(conv30)

        up21 = Conv2DTranspose(filters = self.filters[2], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample21')(conv30)
        conv21 = concatenate([up21, conv20], name='concat21')

        conv21 = self.AadyasConvolutionBlock(conv21, block_level='21', filters = self.filters[2])

        up12 = Conv2DTranspose(filters = self.filters[1], kernel_size = 2, strides = 2, padding='same', name = 'upsample12')(conv21)
        conv12 = concatenate([up12, conv10, conv11], name = 'concat12')
        conv12 = self.skip_path(conv12, block_level = '12', filters = self.filters[1])

        up03 = Conv2DTranspose(filters = self.filters[0], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample03')(conv12)
        conv03 = concatenate([up03, conv00, conv01, conv02], name = 'concat03')
        conv03 = self.AadyasConvolutionBlock(conv03, block_level = '03', filters = self.filters[0])

        conv40 = self.AadyasConvolutionBlock(pool3, block_level = '40', filters = self.filters[4])

        up31 = Conv2DTranspose(filters = self.filters[3], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample31')(conv40)
        conv31 = concatenate([up31, conv30], name = 'concat31')
        conv31 = self.skip_path(conv31, block_level = '31', filters=self.filters[3])

        up22 = Conv2DTranspose(filters = self.filters[2], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample22')(conv31)
        conv22 = concatenate([up22, conv20, conv21], name = 'concat22')
        conv22 = self.skip_path(conv22, block_level = '22', filters = self.filters[2])

        up13 = Conv2DTranspose(filters = self.filters[1], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample13')(conv22)
        conv13 = concatenate([up13, conv10, conv11, conv12], name='concat13')
        conv13 = self.skip_path(conv13, block_level = '13', filters = self.filters[1])

        up04 = Conv2DTranspose(filters = self.filters[0], kernel_size = 2, strides = 2, padding = 'same', name = 'upsample04')(conv13)
        conv04 = concatenate([up04, conv00, conv01, conv02, conv03], name = 'concat04')
        conv04 = self.AadyasConvolutionBlock(conv04, block_level = '04', filters = self.filters[0])

        nested_op_1 = Conv2D(filters = self.num_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding = 'same', name = 'op1')(conv01)

        nested_op_2 = Conv2D(filters = self.num_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding = 'same', name = 'op2')(conv02)

        nested_op_3 = Conv2D(filters = self.num_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding= 'same', name = 'op3')(conv03)

        nested_op_4 = Conv2D(filters = self.num_classes, kernel_size = 1, activation = tf.nn.sigmoid, 
                                  padding = 'same', name = 'op4')(conv04)

        if self.deep_supervision:
            output = [nested_op_1, nested_op_2, nested_op_3, nested_op_4]
        else:
            output = [nested_op_4]

        model = Model(inputs = input_img, outputs = output, name = "UNetPP")

        return model

    
    def skip_path(self, input_tensor, block_level, filters, kernel_size = 3):

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_1')(input_tensor)

        x = Dropout(rate = 0.5, name = 'X' + block_level + '_')(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_2')(x)

        x = Dropout(rate = 0.5, name = 'X' + block_level)(x)

        return x

    def AadyasConvolutionBlock(self, input_tensor, block_level, filters, kernel_size = 3):


        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_1')(input_tensor)

        x = Dropout(rate = 0.5, name = 'X' + block_level + '_')(x)

        x = Conv2D(filters = filters, kernel_size = kernel_size, activation = tf.nn.relu, 
                   padding = 'same', name = 'conv' + block_level + '_2')(x)

        x = Dropout(rate = 0.5, name = 'X' + block_level)(x)

        return x

    
    def CompileAndSummarizeModel(self, model, optimizer='adam'):

        
        model.compile(optimizer = optimizer, 
                      loss = 'binary_crossentropy',
                       metrics = iou
                     )
        
        model.summary()