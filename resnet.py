# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

def resnet_model(image_height, image_width, n_channels, load_wt = "Yes"):
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')

    def identity_block(input_tensor,kernel_size,filters,stage,block):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res'+str(stage)+block+'_branch'
        bn_name_base = 'bn'+str(stage)+block+'_branch'

        x = Conv2D(filters1,(1,1),name=conv_name_base+'2a')(input_tensor)
        x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)

        x = layers.add([x,input_tensor])
        x = Activation('relu')(x)

        return x

    def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
        filters1,filters2,filters3 = filters
        conv_name_base = 'res'+str(stage)+block+'_branch'
        bn_name_base = 'bn'+str(stage)+block+'_branch'

        x = Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
        x = BatchNormalization(axis=3,name=bn_name_base+'2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
        x = BatchNormalization(axis=3,name=bn_name_base+'2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    img_input = Input(shape=(image_height, image_width, n_channels))
    img_padding = ZeroPadding2D(((41,41),(11,11)))(img_input)
    x = ZeroPadding2D((3,3))(img_padding)
    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
    x = BatchNormalization(axis=3,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv_block(x, 3, [64,64,256],stage=2,block='a',strides=(1,1))
    x = identity_block(x,3,[64,64,256],stage=2,block='b')
    x = identity_block(x,3,[64,64,256],stage=2,block='c')

    x = conv_block(x,3,[128,128,512],stage=3,block='a')
    x = identity_block(x,3,[128,128,512],stage=3,block='b')
    x = identity_block(x,3,[128,128,512],stage=3,block='c')
    x = identity_block(x,3,[128,128,512],stage=3,block='d')

    x = conv_block(x,3,[256,256,1024],stage=4,block='a')
    x = identity_block(x,3,[256,256,1024],stage=4,block='b')
    x = identity_block(x,3,[256,256,1024],stage=4,block='c')
    x = identity_block(x,3,[256,256,1024],stage=4,block='d')
    x = identity_block(x,3,[256,256,1024],stage=4,block='e')
    x = identity_block(x,3,[256,256,1024],stage=4,block='f')

    x = conv_block(x,3,[512,512,2048],stage=5,block='a')
    x = identity_block(x,3,[512,512,2048],stage=5,block='b')
    x = identity_block(x,3,[512,512,2048],stage=5,block='c')

    x = AveragePooling2D((7,7),name='avg_pool')(x)
    x = Flatten()(x)

    inp = img_input

    model = Model(inp,x,name='resnet50')

    if load_wt == "Yes":
        model.load_weights(weights_path)

    return model
