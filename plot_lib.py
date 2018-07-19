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

    def on_click_bnext(self, event):
        print("Hello world, details of event token: {}".format(event))
        print("****************************************************************************************************")
        return True

    # link: https://matplotlib.org/examples/widgets/buttons.html
    def plot(self, hair_type="3c"):

        i = self._hair_types.index(hair_type.strip().lower())
        print("Value of i: {}".format(i))
        plt.close()
        plt.figure(figsize=(10,7), clear=True)
        plt.suptitle('Batch plot of hair type {}'.format(hair_type), fontsize=15)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next hair type')
        bprev = Button(axprev, 'Previous hair type')
        # bnext.on_clicked([self.on_click_bnext, self.plot(self._hair_types[(i+1) % self._n])])
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
    def plot_new_figure(self, hair_type="3c"):
        self._plot_new_figure()

    def _plot_new_figure(self, hair_type="3c"):
        i = self._hair_types.index(hair_type)
        for idx, img in enumerate(self._arr[i]):
            if idx % 9 == 0:
                plt.close()
                plt.figure(figsize=(10, 7))
                plt.suptitle('Batch plot of hair type {}'.format(hair_type), fontsize=15)
            plt.subplot(3, 3, (idx%9)+1)
            plt.xlabel('image width')
            plt.ylabel('image height')
            plt.imshow(img)
            # b3c =
            # b4c =
            plt.waitforbuttonpress(-1)
        plt.show()

import algorithm_lib as alib

def fetch_data():
    """
    :return: TBD
    """
    # bin3c = binary image of type 3c
    bin3c, _, _, _, _ = alib.load_preprocess_contours("3c", 200)
    # bin4a, _, _, _, _ = alib.load_preprocess_contours("4a", 200)
    #bin4b, _, _, _, _ = alib.load_preprocess_contours("4b", 200)
    bin4c, _, _, _, _ = alib.load_preprocess_contours("4c", 200)
    arr = [bin3c, bin4c]
    # plt2.Index(arr, ["3c", "4c"]).plot_new_figure()
    Index(arr, ["3c", "4c"]).plot()

if __name__ == '__main__':
    fetch_data()
