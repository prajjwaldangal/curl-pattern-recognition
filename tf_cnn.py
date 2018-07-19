import tensorflow as tf
import numpy as np

import algorithm_lib as alib
import plot_lib as plt2

""""
Our network uses a combination of pixel intensities and weights.

The neural network specifications:
Zero padding = 1 on each edge
Stride = 2 both horizontally and vertically
    1x32x32 - 256C3   -   MP2 -  16C2  -  MP2   - 8C2  - 10N - 4N - 1N (class prediction)
    input     32x16x16  32x8x8  16x4x4  16x2x2  8x1x1
    
# output volume formula:  (Img width−Filter size+2*Padding)/Stride+1

30x30 input images, 3  conv. layer, 2 max. pool layers, 2 fully-conn. layers
#relu after conv layer
loss function = SVM loss
activation function = relu

# segmentation specifications:
    segment only the hair part with high precision
    

07/16/2018 notes for team meeting:
figured out input with tensorflow
initialized some of the architecture specifications

    
"""

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 2
num_filters2 = 16

# Convolutional Layer 3.
filter_size3 = 2
num_filters3 = 8

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

train_path = 'data/train/' # + '3c'/'4a'/'4b' ...
test_path = 'data/test/' # similar to train path
checkpoint_dir = "models/" #

# the matrices to be learnt
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# the biases, also are learnt
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# we need to reduce the 4-dim tensor to 2-dim which can be used as 
# input to the fully-connected layer
def flatten_layer(layer):
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    # i.e. the first dimension will be num_of_elements_
    layer_flat = tf.reshape(layer, [-1, num_features])


# tensor types in tensorflow
# Rank 0 tensors: tf.Variable("Elephant", tf.string), tf.Variable(451/3.14159/2.2+7j, tf.int16, tf.float64,
#                                                                                     tf.complex64) respectively
# Rank 1 tensors: tf.Variable(["Elephant"]/[3.1416, 2.7183]/[2,3,4]/[12+7j, 2-3.5j],
#                                                           tf.string/tf.float32/tf.int32/tf.complex64) respectively
# similarly rank 2 would be list of list, rank 3 would be list of list of list

# a rank 2 tensor with shape [3,4] : [ [1, 2, 3, 4], [2, 4, 6, 8], [-1, -2, -3, -4] ]
# following is a rank 3 tensor with shape [1, 4, 3]:
# [ [
#       [ 1, 2, 3 ],
#       [ 2, 4, 6 ],
#       [ 3, 6, 9 ],
#       [ 5, 10, 15 ]
#  ], [
#       [-1, -2, -3],
#       [-2, -4, -6],
#       [-3, -6, -9],
#       [-5, -10, -15]
#   ] ], more info: https://www.tensorflow.org/guide/tensors
rank_three_tensor = tf.ones([3,4,5])
rank_three_tensor_np = np.ones((3,4,5))
r_two_tf = tf.reshape(rank_three_tensor, [10,-1])
r_two_np = rank_three_tensor_np.reshape((10,6))


# concerned with only the input at this point
x = tf.placeholder(tf.float32, (30, 30), name = 'x')
# tf.train.GradientDescentOptimizer vs tf.train.AdamOptimizer for mnist vs cnn respectively.
session = tf.Session()
session.run(tf.initialize_all_variables())
# try help(tf.placeholder) for code with feeding

def fetch_data():
    """
    :return: TBD
    """
    # bin3c = binary image of type 3c
    bin3c, _, _, _, _ = alib.load_preprocess_contours("3c", 200)
    # bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", 200)
    #bin4b, _, _, _, _ = alib.load_preprocess_contours("4b", 200)
    # bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", 200)
    arr = [bin3c]
    plt2.Index(arr, ["3c", "4c"]).plot()

    # mix
    pass

    feed_dict_train = {x: x_batch,
                       y_true: y_true_batch}

    feed_dict_validate = {x: x_valid_batch,
                          y_true: y_valid_batch}

fetch_data()

# Counter for total number of iterations performed so far.
total_iterations = 0
import time
def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

# help(tf.keras)
# https://www.tensorflow.org/tutorials/deep_cnn
# vs
# https://github.com/rdcolema/tensorflow-image-classification/blob/master/cnn.ipynb

