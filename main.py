import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io
import cv2
import os
import heapq
import scipy.misc

w = 1/5
w_inverse = int(1/w)

DEBUG = True
mixed_image_path = "./sample_debug.png" if DEBUG else "./sample.png"


def load_dataset():
    lfw_dataset_root = "D:\\Docs\\Machine Learning\\Data\\LFW\\lfw_test" if DEBUG else "D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw"
    lfw_dataset = dict()
    lfw_dataset['sobel_x'] = []
    lfw_dataset['sobel_y'] = []
    lfw_dataset['labels'] = []
    _, dirnames, _ = next(os.walk(lfw_dataset_root))
    count = 0
    for face_label in dirnames:
        dirpath, _, filenames = next(os.walk(os.path.join(lfw_dataset_root, face_label)))
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path, 0)
            # TODO: No Need?
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # ksize 1 and 3 both looks good
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # ksize 1 and 3 both looks good
            lfw_dataset['sobel_x'].append(sobel_x)
            lfw_dataset['sobel_y'].append(sobel_y)
            lfw_dataset['labels'].append(face_label)
            del image
            print(filename)
            count += 1
    print("%d faces in the dataset." % count)
    lfw_dataset['len'] = count
    return lfw_dataset


def read_mixed_image():
    original_image = cv2.imread(mixed_image_path, 0)
    image = dict()
    image['sobel_x'] = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
    image['sobel_y'] = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)
    return image


def a_star_search(data):
    # A* search
    candidate_heap = [(0, 0, 1, -1)]     # (score, cumulated_grad_loss, -layer, img_idx)
    came_from = dict()      # (layer, idx) -> (parent_layer, idx)
    best_grad_loss = dict()

    def reconstruct_path(node_tuple, came_from):
        if node_tuple[2] == 1:
            return []
        path = [node_tuple[3]]
        parent = came_from[(node_tuple[2], node_tuple[3])]
        while parent[0] != 1:
            path.append(parent[1])
            parent = came_from[parent]

        return path      # [img_idx]

    def heuristic_cost_estimate(prev_node_tuple, came_from, current_source_idx, data):
        path = reconstruct_path(prev_node_tuple, came_from)
        mixed_image = read_mixed_image()
        reconstructed_image = dict()
        reconstructed_image['sobel_x'] = w_inverse * data['sobel_x'][current_source_idx]
        reconstructed_image['sobel_y'] = w_inverse * data['sobel_y'][current_source_idx]
        for idx in path:
            reconstructed_image['sobel_x'] += w_inverse * data['sobel_x'][idx]
            reconstructed_image['sobel_y'] += w_inverse * data['sobel_y'][idx]

        # Euclidean distance
        loss_x = np.sqrt(np.sum(np.square(mixed_image['sobel_x'] - reconstructed_image['sobel_x'])))
        loss_y = np.sqrt(np.sum(np.square(mixed_image['sobel_y'] - reconstructed_image['sobel_y'])))
        return loss_x + loss_y

    def gradient_loss(residual, source_idx, data):
        # weighted gradient
        loss_x = np.absolute(data['sobel_x'][source_idx]/255 * (w_inverse * residual['sobel_x'] - data['sobel_x'][source_idx]))
        loss_y = np.absolute(data['sobel_x'][source_idx]/255 * (w_inverse * residual['sobel_y'] - data['sobel_y'][source_idx]))
        # TODO: mean or sum?
        return np.sum(loss_x + loss_y)


    def reconstruct_residual_gradient(node_tuple, came_from, data):
        path = reconstruct_path(node_tuple, came_from)
        mixed_image = read_mixed_image()
        for idx in path:
            mixed_image['sobel_x'] = mixed_image['sobel_x'] - data['sobel_x'][idx]
            mixed_image['sobel_y'] = mixed_image['sobel_y'] - data['sobel_y'][idx]
        return mixed_image

    while True:
        current = heapq.heappop(candidate_heap)
        print("current node tuple: ", current)
        if current[2] == -4:
            # end
            return reconstruct_path(current, came_from)

        # TODO: don't forget to subtract gradient
        residual = reconstruct_residual_gradient(current, came_from, data)
        for i in range(data['len']):
            grad_loss = gradient_loss(residual, i, data)
            cumulative_grad_loss = grad_loss + current[1]
            new_layer = current[2]-1
            tentative_score = cumulative_grad_loss + heuristic_cost_estimate(current, came_from, i, data)

            if not(((new_layer, i) in best_grad_loss) and (cumulative_grad_loss >= best_grad_loss[(new_layer, i)])):
                best_grad_loss[(new_layer, i)] = cumulative_grad_loss
                came_from[(new_layer, i)] = (current[2], current[3])

            heapq.heappush(candidate_heap, (tentative_score, cumulative_grad_loss, new_layer, i))


def signal_separation():
    data = load_dataset()
    face_idx_list = a_star_search(data)
    for face_idx in face_idx_list:
        print(data['labels'][face_idx])



signal_separation()

def generate_mixed_image():
    test_data_path = "D:\\Docs\\Machine Learning\\Data\\LFW\\lfw_test"
    img1 = io.imread(os.path.join(test_data_path, "Aaron_Eckhart\\Aaron_Eckhart_0001.jpg"), 0)
    img2 = io.imread(os.path.join(test_data_path, "Tom_Kelly\\Tom_Kelly_0001.jpg"), 0)
    img3 = io.imread(os.path.join(test_data_path, "Valdas_Adamkus\\Valdas_Adamkus_0001.jpg"), 0)
    img4 = io.imread(os.path.join(test_data_path, "Aaron_Peirsol\\Aaron_Peirsol_0001.jpg"), 0)
    img5 = io.imread(os.path.join(test_data_path, "Romain_Duris\\Romain_Duris_0001.jpg"), 0)

    merged_img = w*img1 + w*img2 + w*img3 + w*img4 + w*img5
    merged_img = merged_img.astype(np.uint8)
    io.imshow(merged_img)

    scipy.misc.imsave("./sample_debug.png", merged_img)
    io.show()


def experiment():
    w = 1/5

    # sample = io.imread("./sample.png", as_grey=True)
    # sample = cv2.imread("./sample.png", 0)
    # sample = cv2.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg", 0)




    # print(sample.shape)
    # cv2.imshow("mixed image", sample)

    # laplacian = cv2.Laplacian(sample, cv2.CV_64F, ksize=3, scale=1)
    # laplacian = cv2.convertScaleAs(laplacian)
    # cv2.threshold(laplacian, )
    # print(laplacian)
    # laplacian = np.uint8(laplacian)

    # sobelx = cv2.Sobel(sample, cv2.CV_64F, 1, 0, ksize=3)       # ksize 1 and 3 both looks good
    # sobelx = cv2.convertScaleAbs(w * sobelx)
    # sobelx = cv2.convertScaleAbs(sobelx)
    # sobely = cv2.Sobel(sample, cv2.CV_64F, 0, 1, ksize=3)       # ksize 1 and 3 both looks good
    # sobely = cv2.convertScaleAbs(w * sobelx)
    # sobely = cv2.convertScaleAbs(sobely)

    # sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # print(sobel)
    # abs_sobelx = np.absolute(sobelx)
    # sobelx_8u = np.uint8(abs_sobelx)

    # cv2.imshow("lap", laplacian)

    # io.imshow(laplacian)
    # io.imshow(sobel)


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

    # pic1 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg")
    # pic2 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Johnny_Tapia\\Johnny_Tapia_0003.jpg")
    # pic3 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Joshua_Perper\\Joshua_Perper_0001.jpg")
    # pic4 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Michael_McNeely\\Michael_McNeely_0001.jpg")
    # pic5 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Shannon_OBrien\\Shannon_OBrien_0001.jpg")
    #
    # pic = w*pic1 + w*pic2 + w*pic3 + w*pic4 + w*pic5
    # pic = pic.astype(np.uint8)
    # io.imshow(pic)
    #
    # io.show()
    # cv2.waitKey(0)

# experiment()
# generate_mixed_image()
