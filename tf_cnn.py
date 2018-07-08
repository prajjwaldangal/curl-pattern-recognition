import tensorflow as tf

""""
Our network uses a combination of pixel intensities and weights.

The neural network specifications:
Zero padding = 1 on each edge
Stride = 2 both horizontally and vertically
    1x32x32 - 32C3   -   MP2 -  16C2  -  MP2   - 8C2  - 10N - 4N - 1N (class prediction)
    input     32x16x16  32x8x8  16x4x4  16x2x2  8x1x1
# output volume formula:  (Img width−Filter size+2*Padding)/Stride+1

30x30 input images, 3  conv. layer, 2 max. pool layers, 2 fully-conn. layers
#relu after conv layer
loss function = SVM loss
activation function = relu

# segmentation specifications:
    segment only the hair part with high precision
    
"""

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale, 3 for RGB.
num_channels = 1

# image dimensions (only squares for now)
#img_size = 128
img_size = 30

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['3c', '4a', '4b', '4c']
num_classes = len(classes)

# batch size
batch_size = 32

# validation split
validation_size = .16

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'data/train/'
test_path = 'data/test/test/'
checkpoint_dir = "models/"

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache



# we need to reduce the 4-dim tensor to 2-dim which can be used as 
# input to the fully-connected layer
def flatten_layer(layer):
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])
    # partial derivative

# https://www.tensorflow.org/tutorials/deep_cnn
# vs
# https://github.com/rdcolema/tensorflow-image-classification/blob/master/cnn.ipynb

