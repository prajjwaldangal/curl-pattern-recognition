import tensorflow as tf

# we need to reduce the 4-dim tensor to 2-dim which can be used as 
# input to the fully-connected layer

def 

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