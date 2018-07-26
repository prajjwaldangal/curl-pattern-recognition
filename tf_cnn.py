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
img_size = 50

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


# concerned with only the input at this point
x = tf.placeholder(tf.float32, (30, 30), name = 'x')
# tf.train.GradientDescentOptimizer vs tf.train.AdamOptimizer for mnist vs cnn respectively.
session = tf.Session()

# session.run(tf.initialize_all_variables())  --> will be removed after 2017-03-02
# try help(tf.placeholder) for code with feeding

def mix(ls):
    """
    :param ls: list of different hair type images
    :return: train and validation image sets
    """
    if ls == [] or ls[0] == [] or ls[1] == []:
        return
    n_hair_types = len(ls)
    n_imgs = len(ls[0]) # number of images in each hair_type array
    img_height = len(ls[0][0])
    img_width = len(ls[0][0][0])

    # the following code turns a x b x c x d arr into l x c x d where l = a x b, mixes them and separates
    # into train and validation sets
    train_arr = []
    valid_arr = []
    # arr = np.zeros((n_hair_types * n_imgs, img_height, img_width))
    for hair_matrices in ls:
        l = int((1-validation_size)*n_imgs)+1
        for idx, img in enumerate(hair_matrices[:l]):
            train_arr.append(img)
        for idx, img in enumerate(hair_matrices[l:]):
            valid_arr.append(img)
    # mix
    print("Shuffling...")
    alib.dots(5)
    t = len(train_arr)
    v = len(valid_arr)
    train_rtrn = np.zeros((t, img_height, img_width))
    valid_rtrn = np.zeros((v, img_height, img_width))
    train_seen = {}
    valid_seen = {}
    train_cnt = 0
    valid_cnt = 0
    done = False
    while not done:
        train_rand_int = random.randint(0, t-1)
        valid_rand_int = random.randint(0, v-1)
        if not train_rand_int in train_seen:
            train_rtrn = np.insert(train_rtrn, train_rand_int, train_arr[train_rand_int], 0)
            train_rtrn = np.delete(train_rtrn, train_rand_int+1, 0)
            train_cnt += 1

        if not valid_rand_int in valid_seen:
            valid_rtrn = np.insert(valid_rtrn, valid_rand_int, valid_arr[valid_rand_int], 0)
            valid_rtrn = np.delete(valid_rtrn, valid_rand_int+1, 0)
            valid_cnt += 1

        done = train_cnt < t and valid_cnt < v
    # compare ls and train and valid rtrn
    def assure(ls, arr1, arr2):
        print("Length of ls: {}\nLength of train: {}, of valid: {}".format(len(ls), len(arr1), len(arr2)))
        # can put more tests for assurance progressively

    assure(ls[0]+ls[1], train_rtrn, valid_rtrn)
    return train_rtrn, valid_rtrn


def fetch_data(segmented_thus_less=False, n=10, dim=(30, 30)):
    """

    :param segmented_thus_less: if true, fetch segmented else fetch unsegmented
    :param n: number of images to fetch in total
    :param dim: load images with dimension dim
    :return:
    """
    # TO-DO: (1) Segment 3c (2) Put into folders. Same for 4b
    # bin4a = binary image of type 4a
    # bin, grays, originals, conts_ls, canny = alib.load_preprocess_contours("4a", 50, (50, 50), ...
    if not segmented_thus_less:
        bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", n, dim, segmented=False)
        bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", n, dim, segmented=False)
    else:
        bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", n, dim)
        bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", n, dim)
    # dim(bin4a) = 10x30x30

    arr = [bin4a, bin4c]
    # learning rate check after each epoch
    # backpropagation for each row


    # Manager is a design smell
    # name classes in terms of the functional requirement it fulfills
    # Mediator pattern: Objects no longer communicate directly with each other, but instead communicate through the
    #                                                                                           mediator.
    # https://softwareengineering.stackexchange.com/questions/350142/how-can-i-manage-the-code-base-of-significantly-complex-software

    # use 1 out of 10 as validation, make general

    x_train, x_valid = mix(arr)
    
    # x_batch = arr[0][:train_per*len(arr)]

    feed_dict_train = {x: x_batch,
                       y_true: y_true_batch}

    feed_dict_validate = {x: x_valid_batch,
                          y_true: y_valid_batch}

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
# vs
# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/#more-452
# vs
# https://datascience.stackexchange.com/questions/24511/why-should-the-data-be-shuffled-for-machine-learning-tasks

if __name__ == '__main__':
    fetch_data(False)
