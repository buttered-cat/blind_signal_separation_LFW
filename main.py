import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io
import cv2
import os


lfw_dataset_root = "D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw"
lfw_dataset = dict()
lfw_dataset['sobel_x'] = []
lfw_dataset['sobel_y'] = []
lfw_dataset['labels'] = []

def load_dataset():
    _, dirnames, _ = next(os.walk(lfw_dataset_root))
    count = 0
    for face_label in dirnames:
        dirpath, _, filenames = next(os.walk(os.path.join(lfw_dataset_root, face_label)))
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path, 0)
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # ksize 1 and 3 both looks good
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # ksize 1 and 3 both looks good
            lfw_dataset['sobel_x'].append(sobel_x)
            lfw_dataset['sobel_y'].append(sobel_y)
            lfw_dataset['labels'].append(face_label)
            del image
            print(filename)
            count += 1
    print(count)


def signal_separation():
    load_dataset()

signal_separation()

def experiment():
    w = 1/5

    # sample = io.imread("./sample.png", as_grey=True)
    sample = cv2.imread("./sample.png", 0)
    # sample = cv2.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg", 0)
    print(sample.shape)
    # cv2.imshow("mixed image", sample)

    # laplacian = cv2.Laplacian(sample, cv2.CV_64F, ksize=3, scale=1)
    # laplacian = cv2.convertScaleAs(laplacian)
    # cv2.threshold(laplacian, )
    # print(laplacian)
    # laplacian = np.uint8(laplacian)

    sobelx = cv2.Sobel(sample, cv2.CV_64F, 1, 0, ksize=3)       # ksize 1 and 3 both looks good
    # sobelx = cv2.convertScaleAbs(w * sobelx)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(sample, cv2.CV_64F, 0, 1, ksize=3)       # ksize 1 and 3 both looks good
    # sobely = cv2.convertScaleAbs(w * sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    print(sobel)
    # abs_sobelx = np.absolute(sobelx)
    # sobelx_8u = np.uint8(abs_sobelx)

    # cv2.imshow("lap", laplacian)

    # io.imshow(laplacian)
    io.imshow(sobel)


    # img = cv2.imread("./sample.png", 0)
    # print(img.shape)
    # print(type(img))
    # io.imshow(img)



    # plt.imshow(sample)
    # pca = PCA(n_components=100)
    # pca.fit(sample)
    # sample_pca = pca.fit_transform(sample)
    #
    # print(sample_pca.shape)
    # sample_pca_restored = pca.inverse_transform(sample_pca)
    # io.imshow(sample_pca_restored)

    # ica = FastICA()
    # S = ica.fit_transform(sample)
    # plt.show()

    # pic1 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg", as_grey=True)
    # pic2 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Johnny_Tapia\\Johnny_Tapia_0003.jpg", as_grey=True)
    # pic3 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Joshua_Perper\\Joshua_Perper_0001.jpg", as_grey=True)
    # pic4 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Michael_McNeely\\Michael_McNeely_0001.jpg", as_grey=True)
    # pic5 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Shannon_OBrien\\Shannon_OBrien_0001.jpg", as_grey=True)
    #
    # pic = w*pic1 + w*pic2 + w*pic3 + w*pic4 + w*pic5
    # io.imshow(pic)

    io.show()
    # cv2.waitKey(0)

# experiment()
