# Author: Jingxiao Gu
# Baidu Account: Seigato
# Description: Unet Base Network for Lane Segmentation Competition

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None, bn=True, bias_attr=False):
    conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=bias_attr,
            param_attr=ParamAttr(initializer=MSRA()))
    if bn == True:
          conv = fluid.layers.batch_norm(input=conv, act=act)
    return conv

def conv_layer(input, num_filters, filter_size, stride=1, groups=1, act=None):
    conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=act,
            bias_attr=ParamAttr(initializer=MSRA()),
            param_attr=ParamAttr(initializer=MSRA()))
    return conv

def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out or stride != 1:
        return conv_bn_layer(input, ch_out, 1, stride)
    else:
        return input

def bottleneck_block(input, num_filters, stride):
    conv_bn = conv_bn_layer(input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv_bn = conv_bn_layer(input=conv_bn, num_filters=num_filters, filter_size=3, stride=stride, act=None)
    short_bn = shortcut(input, num_filters, stride)
    return fluid.layers.elementwise_add(x=short_bn, y=conv_bn, act='relu')

def encoder_block(input, encoder_depths, encoder_filters, block):
    conv_bn = input
    for i in range(encoder_depths[block]):
        conv_bn = bottleneck_block(input=conv_bn, num_filters=encoder_filters[block], stride=2 if i == 0 and block != 0 else 1)
    print("| Encoder Block", block, conv_bn.shape)
    return conv_bn

def decoder_block(input, concat_input, decoder_depths, decoder_filters, block):
    deconv_bn = input
    deconv_bn = fluid.layers.resize_bilinear(input=deconv_bn, out_shape=(deconv_bn.shape[2] * 2, deconv_bn.shape[3] * 2))
    deconv_bn = bottleneck_block(input=deconv_bn, num_filters=decoder_filters[block], stride=1)

    concat_input = conv_bn_layer(input=concat_input, num_filters=concat_input.shape[1] // 2, filter_size=1, act='relu')

    deconv_bn = fluid.layers.concat([deconv_bn, concat_input], axis=1)
    for i in range(decoder_depths[block]):
        deconv_bn = bottleneck_block(input=deconv_bn, num_filters=decoder_filters[block], stride=1)
    print('| Decoder Block', block, deconv_bn.shape)
    return deconv_bn

def unet_base(img, label_number, img_size):
    print("| Build Custom-Designed Resnet-Unet:")
    encoder_depth = [3, 4, 6, 4]
    encoder_filters = [64, 128, 256, 512]
    decoder_depth = [4, 3, 3, 2]
    decoder_filters = [256, 128, 64, 32]
    print('| Input Image Data', img.shape)
    """ 
    Encoder
    """
    # Start Conv
    start_conv = conv_bn_layer(input=img, num_filters=32, filter_size=3, stride=2, act='relu')
    start_conv = conv_bn_layer(input=start_conv, num_filters=32, filter_size=3, stride=1, act='relu')
    start_pool = fluid.layers.pool2d(input=start_conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
    print('| Start Convolution', start_conv.shape)

    conv0 = encoder_block(start_pool, encoder_depth, encoder_filters, block=0)
    conv1 = encoder_block(conv0, encoder_depth, encoder_filters, block=1)
    conv2 = encoder_block(conv1, encoder_depth, encoder_filters, block=2)
    conv3 = encoder_block(conv2, encoder_depth, encoder_filters, block=3)

    """ 
    Decoder
    """
    decode_conv1 = decoder_block(conv3, conv2, decoder_depth, decoder_filters, block=0)
    decode_conv2 = decoder_block(decode_conv1, conv1, decoder_depth, decoder_filters, block=1)
    decode_conv3 = decoder_block(decode_conv2, conv0, decoder_depth, decoder_filters, block=2)
    decode_conv4 = decoder_block(decode_conv3, start_conv, decoder_depth, decoder_filters, block=3)

    """ 
    Output Coder
    """
    decode_conv5 = fluid.layers.resize_bilinear(input=decode_conv4, out_shape=img_size)
    decode_conv5 = bottleneck_block(input=decode_conv5, num_filters=32, stride=1)
    decode_conv5 = bottleneck_block(input=decode_conv5, num_filters=16, stride=1)
    logit = conv_layer(input=decode_conv5, num_filters=label_number, filter_size=1, act=None)
    print("| Output Predictions:", logit.shape)
    # logit = fluid.layers.resize_bilinear(input=logit, out_shape=(3384, 1020))
    print("| Final Predictions:", logit.shape)

    return logit