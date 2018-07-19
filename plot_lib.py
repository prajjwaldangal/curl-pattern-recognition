# from algorithm_lib import load_preprocess_contours
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

ls2 = []

# TO-DO: refactor into MVC
class Index(object):

    def __init__(self, arr, hair_types):
        self._arr = arr
        self._hair_types = hair_types

    # link: https://matplotlib.org/examples/widgets/buttons.html
    def plot(self):
        for hair_type in self._hair_types:
            plt.figure(figsize=(10,7), clear=True)
            self._plot(hair_type)


    def next_hair_type(self, event):
        pass

    def _plot(self, hair_type):
        i = self._hair_types.index(hair_type.strip().lower())
        for idx, img in enumerate(self._arr[i]):
            if idx % 9 == 0:
                # plt.clear()
                plt.suptitle('Batch plot of hair type {}'.format(self._hair_types[i]), fontsize=15)
            plt.subplot(3, 3, (idx%9)+1)
            plt.xlabel('image width')
            plt.ylabel('image height')
            plt.imshow(img)
            plt.waitforbuttonpress(-1)
        plt.show()
