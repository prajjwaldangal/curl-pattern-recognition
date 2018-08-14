import tensorflow as tf
import numpy as np
import random

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
fc_size = 32             # Number of neurons in fully-connected layer.

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
# classes = ['3c', '4a', '4b', '4c']
classes = ['4a']
num_classes = len(classes)

# batch size
BATCH_SIZE = 8

# validation split
validation_size = .14

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

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

# session.run(tf.initialize_all_variables())  --> will be removed after 2017-03-02
# try help(tf.placeholder) for code with feeding

def random_mix(ls, hair_types):
    """
    :param ls: list of different hair type images, num_classes x num_examples_of_each_class x img_height x img_width
    :return: train and validation image sets along with their corresponding labels
    """
    if ls == [] or ls[0] == [] or ls[1] == []:
        return
    n_hair_types = len(ls)
    n_imgs = len(ls[0]) # number of images in each hair_type array
    img_height = len(ls[0][0])
    img_width = len(ls[0][0][0])
    # the following code turns a x b x c x d arr into l x c x d where l = a x b, mixes them and separates
    # into train and validation sets
    train_x = []
    valid_x = []
    train_y = []
    valid_y = []
    # arr = np.zeros((n_hair_types * n_imgs, img_height, img_width))
    for i, hair_matrices in enumerate(ls):
        l = int((1-validation_size)*n_imgs)+1
        # label=hair_types[i]
        for idx, img in enumerate(hair_matrices[:l]):
            train_x.append(img)
            train_y.append(i)
        for idx, img in enumerate(hair_matrices[l:]):
            valid_x.append(img)
            valid_y.append(i)
    # mix
    print("Shuffling...")
    alib.dots(5)
    t = len(train_x)
    v = len(valid_x)
    train_rtrn_x = np.zeros((t, img_height, img_width))
    train_rtrn_y = np.zeros((t))
    valid_rtrn_x = np.zeros((v, img_height, img_width))
    valid_rtrn_y = np.zeros((v))
    train_seen = {}
    valid_seen = {}
    train_cnt = 0
    valid_cnt = 0
    done = False
    while not done:
        train_rand_int = random.randint(0, t-1)
        valid_rand_int = random.randint(0, v-1)
        if not train_rand_int in train_seen:
            # insert delete x
            train_rtrn_x = np.insert(train_rtrn_x, train_rand_int, train_x[train_rand_int], 0)
            train_rtrn_x = np.delete(train_rtrn_x, train_rand_int+1, 0)
            # insert delete y
            train_rtrn_y = np.insert(train_rtrn_y, train_rand_int, train_y[train_rand_int], 0)
            train_rtrn_y = np.delete(train_rtrn_y, train_rand_int+1, 0)
            train_cnt += 1

        if not valid_rand_int in valid_seen:
            # same as above: insert delete x
            valid_rtrn_x = np.insert(valid_rtrn_x, valid_rand_int, valid_x[valid_rand_int], 0)
            valid_rtrn_x = np.delete(valid_rtrn_x, valid_rand_int+1, 0)
            # insert delete y
            valid_rtrn_y = np.insert(valid_rtrn_y, valid_rand_int, valid_y[valid_rand_int], 0)
            valid_rtrn_y = np.delete(valid_rtrn_y, valid_rand_int+1, 0)
            valid_cnt += 1

        done = train_cnt < t and valid_cnt < v
    # compare ls and train and valid rtrn
    def assure(ls, train_x, train_y, valid_x, valid_y):
        print("\n")
        print("Printing assurance info")
        alib.dots(5)
        print("\n")
        print("Shape of ls: {}\nShape of train_x: {}\nShape of train_y: {}\nShape of valid_x: {}\n"
              "Shape of valid_y: {}".format((len(ls), len(ls[0]), len(ls[0][0])), train_x.shape, train_y.shape, valid_x.shape,
                                            valid_y.shape))

    # can put more tests for assurance progressively
    assure(ls[0]+ls[1], train_rtrn_x, train_rtrn_y, valid_rtrn_x, valid_rtrn_y)
    return [train_rtrn_x, train_rtrn_y], [valid_rtrn_x, valid_rtrn_y]

# concerned with only the input at this point
# dim = len_hair_type * BATCH_SIZE
X = tf.placeholder(tf.float32, (num_classes * BATCH_SIZE, 30, 30), name = 'X')
Y = tf.placeholder(tf.int8, (num_classes * BATCH_SIZE), name='Y')
# tf.train.GradientDescentOptimizer vs tf.train.AdamOptimizer for mnist vs cnn respectively.
session = tf.Session()

def fetch_data(segmented_thus_less=False, dim=(img_size, img_size)):
    """

    :param segmented_thus_less: if true, fetch segmented else fetch unsegmented
    :param n: number of images to fetch in total
    :param dim: load images with dimension dim, different than dim of placeholder X & Y
    :return:
    """
    # global
    # TO-DO: (1) Segment 3c (2) Put into folders. Same for 4b
    # bin4a = binary image of type 4a
    # bin, grays, originals, conts_ls, canny = alib.load_preprocess_contours("4a", 50, (50, 50), ...
    if not segmented_thus_less:
        segmented = False
    else:
        segmented = True
    
    bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", BATCH_SIZE, dim, segmented=segmented)
    bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", BATCH_SIZE, dim, segmented=segmented)
    # dim(bin4a) = 10x30x30

    # NOTE: always send in ascending order
    arr = [bin4a, bin4c]
    # learning rate check after each epoch
    # backpropagation for each row


    # name classes in terms of the functional requirement it fulfills
    # Mediator pattern: Objects no longer communicate directly with each other, but instead communicate through the
    #                                                                                           mediator.
    # https://softwareengineering.stackexchange.com/questions/350142/how-can-i-manage-the-code-base-of-significantly-complex-software

    # use 1 out of 10 as validation, make general

    #
    # more than 10 hidden layers

    train, valid = random_mix(arr, ["4a", "4c"]) # note hair type is always [3c, 4a, 4b, 4c], [0, 1, 2, 3] order
    train_batch_x, train_batch_y = train
    valid_batch_x, valid_batch_y = valid

    
    # x_train_batch is feed into x which is a placeholder
    feed_dict_train = {X: train_batch_x,
                       Y: train_batch_y}

    feed_dict_validate = {X: valid_batch_x,
                          Y: valid_batch_y}

    session.run(optimize, feed_dict=feed_dict_train)

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

def fetch_data_single(hair_type, segmented_thus_less=True, dim=(img_size, img_size)):
    """

    :param segmented_thus_less: get segmented images or no
    :param dim: the dimension of input images
    :param hair_type:   hair_type [3c, 4a, 4b, 4c]
    :return:
    """
    if not segmented_thus_less:
        segmented = False
    else:
        segmented = True

    bin4a, _, _, _, _ = alib.load_preprocess_contours(hair_type, BATCH_SIZE, dim, segmented=segmented)
    idx = int((1-validation_size)*len(bin4a))
    train_batch_x = bin4a[:idx]
    train_batch_y = np.zeros((len(train_batch_x)))
    valid_batch_x = bin4a[idx:]
    valid_batch_y = np.zeros((len(valid_batch_x)))
    feed_dict_train = {X: train_batch_x,
                       Y: train_batch_y}

    feed_dict_validate = {X: valid_batch_x,
                          Y: valid_batch_y}
    session.run(optimizer, feed_dict=feed_dict_train)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

# help(tf.keras)
# https://www.tensorflow.org/tutorials/deep_cnn
# vs
# https://github.com/rdcolema/tensorflow-image-classification/blob/master/cnn.ipynb
# vs
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/#more-452
# vs
# https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks
# https://www.tensorflow.org/tutorials/images/image_recognition

if __name__ == '__main__':
    fetch_data_single("4a", False)
