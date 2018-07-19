# from algorithm_lib import load_preprocess_contours
import matplotlib.pyplot as plt

## batch plot
def plotting(ls, hair_type):
    for idx, img in enumerate(ls):

        if idx % 25 == 0:
            plt.close()
            plt.figure(figsize=(10, 7) )
            plt.xlabel('width')
            plt.title('Batch plot of hair type {}'.format(hair_type))
            plt.ylabel('height')
            continue
        plt.subplot(5,5,(idx%25)+1)
        plt.imshow(img)
        plt.waitforbuttonpress(-1)
    plt.show()

