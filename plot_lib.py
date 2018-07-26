# from algorithm_lib import load_preprocess_contours
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

ls2 = []

# TO-DO: refactor into MVC
class Index(object):

    def __init__(self, arr, hair_types):
        self._arr = arr
        self._hair_types = hair_types
        self._n = len(hair_types)
        self._i = 0

    def foo_bnext(self):
        self._i += 1
        self.plot_batch()

    def foo_bprev(self):
        self._i -= 1
        self.plot_batch()

    def plot_batch(self):
        fig = plt.figure(figsize=(10,7))
        # plot just the first hair type starting out
        plt.suptitle("Batch plot of hair type {}".format(self._hair_types[self._i % self._n]))
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next hair type')
        bprev = Button(axprev, 'Previous hair type')
        for idx, img in enumerate(self._arr[self._i % self._n]):
            plt.subplot(2,5,(idx%10)+1, facecolor='w')
            # if ValueError: num must be 1 <= num <= 10, not 0, check plt.subplot
            # if image data cannot be converted to float, enter self._arr instead of self._hair_types
            ax = fig.add_subplot(2,5, (idx%10)+1)
            ax.set_facecolor("w")
            plt.xlabel('image width')
            plt.ylabel('image height')
            plt.imshow(img)
            plt.waitforbuttonpress(-1)
        # bnext.on_clicked(self.foo_bnext())
        # bprev.on_clicked(se)
        plt.show()

import algorithm_lib as alib

def fetch_data(unsegmented_thus_more=True):
    """
    :return: TBD
    """
    # TO-DO: (1) Segment 3c (2) Put into folders. Same for 4b
    # bin4a = binary image of type 4a
    # bin, grays, originals, conts_ls, canny = alib.load_preprocess_contours("4a", 50, (50, 50), ...
    if unsegmented_thus_more:
        bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", 50, (50, 50), segmented=False)
        bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", 50, (50, 50), segmented=False)
    else:
        bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", 10, (50, 50))
        bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", 10, (50, 50))
    arr = [bin4a, bin4c]
    index = Index(arr, ["4a", "4c"])
    index.plot_batch()


# if __name__ == '__main__':
    # fetch_data(False)

"""
def _plot_batch(self, hair_type="3c"):
        i = self._hair_types.index(hair_type)
        for idx, img in enumerate(self._arr[i]):
            if idx % 9 == 0:
                plt.close()
                plt.figure(figsize=(10, 7))
                plt.suptitle('Plot of hair type {}'.format(hair_type), fontsize=15)
                axprev = plt.axes([0.65, 0.05, 0.15, 0.075])
                axnext = plt.axes([0.81, 0.05, 0.15, 0.075])
                bnext = Button(axnext, 'Next hair type')
                bprev = Button(axprev, 'Prev hair type')
            plt.subplot(3, 3, (idx%9)+1)
            plt.xlabel('image width')
            plt.ylabel('image height')
            plt.imshow(img)
            # b3c =
            # b4c =
            plt.waitforbuttonpress(-1)
        plt.show()

    # 50  -->  train on 49 and test on 1
    # categorization of methods --> shape-based, textures-based
    # one segment only segmentation, one segment only recognition
    # introduction, methodology, results, discussion, questions

    # plots first 16 images of a given hair type, navigate to other hair types as well
    def plot_batch(self, hair_type, n):
        i = self._hair_types.index(hair_type)
        plt.figure(figsize=(10,7))
        plt.suptitle('Batch plot of hair type {}'.format(hair_type), fontsize=15)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next hair type')
        bprev = Button(axprev, 'Previous hair type')
        bnext.on_clicked(self.plot_batch(self._hair_types[(i+1) % self._n], n=16))
        for idx, img in enumerate(self._arr[i]):
            if idx == n:
                break
            plt.subplot(4, 4, (idx%n)+1)
            plt.xlabel('image width')
            plt.ylabel('image height')
            plt.imshow(img)
            # plt.waitforbuttonpress(-1)
        # add bnext
        plt.show()

    # link: https://matplotlib.org/examples/widgets/buttons.html
    # bnext support not fully functional
    def plot2(self, hair_type="3c"):

        i = self._hair_types.index(hair_type.strip().lower())
        print("Value of i: {}".format(i))
        plt.close()
        plt.figure(figsize=(10,7), clear=True)
        plt.suptitle('Batch plot of hair type {}'.format(hair_type), fontsize=15)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next hair type')
        bprev = Button(axprev, 'Previous hair type')
        bnext.on_clicked(self.plot(self._hair_types[(i+1) % self._n]))
        for idx, img in enumerate(self._arr[i]):
            print(bnext)
            plt.subplot(3, 3, (idx%9)+1)
            plt.xlabel('image width')
            plt.ylabel('image height')
            plt.imshow(img)
            plt.waitforbuttonpress(-1)
        plt.show()
    # bnext1: Address: 0x13ba10668  Axes(0.81,0.05;0.1x0.075)
    # bnext2: Address: 0x17cf65780  Axes(0.81,0.05;0.1x0.075)
    # bnext3: Address: 0x13806d390  Axes(0.81,0.05;0.1x0.075)
"""
