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

DEBUG = False
mixed_image_path = "./sample_debug.png" if DEBUG else "./sample.png"


def load_dataset():
    lfw_dataset_root = "D:\\Docs\\Machine Learning\\Data\\LFW\\lfw_test" if DEBUG else "D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw"
    original_image = w_inverse * cv2.imread(mixed_image_path).astype(np.uint16)
    image_cond = np.clip(original_image, None, 255.)
    del original_image
    lfw_dataset = dict()
    lfw_dataset['x'] = []
    lfw_dataset['y'] = []
    lfw_dataset['labels'] = []
    lfw_dataset['files'] = []
    _, dirnames, _ = next(os.walk(lfw_dataset_root))
    count = 0
    for face_label in dirnames:
        dirpath, _, filenames = next(os.walk(os.path.join(lfw_dataset_root, face_label)))
        for filename in filenames:
            image_path = os.path.join(dirpath, filename)
            image = cv2.imread(image_path)

            cond = image <= image_cond
            if np.all(cond):
                # TODO: No Need?
                del image
                image = cv2.imread(image_path, 0)
                sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # ksize 1 and 3 both looks good
                sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # ksize 1 and 3 both looks good
                lfw_dataset['x'].append(sobel_x)
                lfw_dataset['y'].append(sobel_y)
                lfw_dataset['labels'].append(filename)
                lfw_dataset['files'].append(image_path)
                print(filename)
                count += 1
            del image
    print("%d faces in the dataset." % count)
    lfw_dataset['len'] = count
    return lfw_dataset


def read_mixed_image():
    original_image = cv2.imread(mixed_image_path, 0)
    image = dict()
    image['x'] = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
    image['y'] = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)
    return image





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
    reconstructed_image['x'] = w * data['x'][current_source_idx]
    reconstructed_image['y'] = w * data['y'][current_source_idx]
    for idx in path:
        reconstructed_image['x'] += w * data['x'][idx]
        reconstructed_image['y'] += w * data['y'][idx]

    # Euclidean distance
    loss_x = np.sqrt(np.sum(np.square(mixed_image['x'] - reconstructed_image['x'])))
    loss_y = np.sqrt(np.sum(np.square(mixed_image['y'] - reconstructed_image['y'])))
    return loss_x + loss_y

def gradient_loss(residual, source_idx, data):
    # Euclidean distance
    max_abs_data_x = np.max(np.absolute(data['x'][source_idx]))
    max_abs_data_y = np.max(np.absolute(data['y'][source_idx]))

    loss_x = np.sqrt(np.sum(np.square(data['x'][source_idx]/max_abs_data_x * (residual['x'] - w * data['x'][source_idx]))))
    loss_y = np.sqrt(np.sum(np.square(data['y'][source_idx]/max_abs_data_y * (residual['y'] - w * data['y'][source_idx]))))
    # TODO: mean or sum?
    return loss_x + loss_y


def reconstruct_residual_gradient(node_tuple, came_from, data):
    path = reconstruct_path(node_tuple, came_from)
    mixed_image = read_mixed_image()
    for idx in path:
        # mixed_image['x'] = mixed_image['x'] - w * data['x'][idx]/255 * data['x'][idx]
        mixed_image['x'] = mixed_image['x'] - w * data['x'][idx]
        mixed_image['y'] = mixed_image['y'] - w * data['y'][idx]
    return mixed_image


def is_matched(node_tuple, came_from, data):
    original_image = cv2.imread(mixed_image_path, 0)
    path = reconstruct_path(node_tuple, came_from)

    reconstructed_image = None
    
    for idx in path:
        if reconstructed_image is None:
            reconstructed_image = w * cv2.imread(data['files'][idx], 0)
        else:
            reconstructed_image += w * cv2.imread(data['files'][idx], 0)

    loss = np.sqrt(np.sum(np.square(reconstructed_image - original_image)))
    
    if loss == 0:
        return True
    return False



def a_star_search(data):
    # A* search
    candidate_heap = [(0, 0, 1, -1)]     # (score, cumulated_grad_loss, -layer, img_idx)
    came_from = dict()      # (layer, idx) -> (parent_layer, idx)
    best_grad_loss = dict()

    solns = []
    soln_num = 0

    while True:
        current = heapq.heappop(candidate_heap)
        print("current node tuple: ", current)
        if current[2] == -4:
            # end
            solns.append(reconstruct_path(current, came_from))
            soln_num += 1
            if soln_num >= 1:
                return solns

        else:
            # TODO: don't forget to subtract gradient
            residual = reconstruct_residual_gradient(current, came_from, data)
            for i in range(data['len']):
                grad_loss = gradient_loss(residual, i, data)
                # cumulative_grad_loss = grad_loss + current[1]
                new_layer = current[2]-1
                # tentative_score = cumulative_grad_loss + heuristic_cost_estimate(current, came_from, i, data)

                cumulative_grad_loss = grad_loss
                tentative_score = cumulative_grad_loss


                if not(((new_layer, i) in best_grad_loss) and (cumulative_grad_loss >= best_grad_loss[(new_layer, i)])):
                    best_grad_loss[(new_layer, i)] = cumulative_grad_loss
                    came_from[(new_layer, i)] = (current[2], current[3])
                    heapq.heappush(candidate_heap, (tentative_score, cumulative_grad_loss, new_layer, i))
                else:
                    continue


def greedy():
    # NOT MODIFIED!
    data = load_dataset()
    residual = read_mixed_image()
    best_match = [0] * 5
    for layer in range(5):
        best_match[layer] = -1
        best_loss = gradient_loss(residual, data['len'] - 1, data)
        for i in range(data['len'] - 1):
            current_loss = gradient_loss(residual, i, data)
            if current_loss < best_loss:
                best_loss = current_loss
                best_match[layer] = i

        residual['x'] = residual['x'] - data['x'][best_match[layer]]
        residual['y'] = residual['y'] - data['y'][best_match[layer]]

    for face_idx in best_match:
        print(data['labels'][face_idx])

# greedy()


def signal_separation():
    data = load_dataset()
    possible_solns = a_star_search(data)
    soln_num = 0
    for face_idx_list in possible_solns:
        soln_num += 1
        print("possibility %d:" % soln_num)
        for face_idx in face_idx_list:
            print(data['labels'][face_idx])
        print('\n')






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
    sample = cv2.imread("./sample1.png", 0)

    # print(sample.shape)
    # cv2.imshow("mixed image", sample)

    laplacian = cv2.Laplacian(sample, cv2.CV_64F, ksize=3, scale=1)
    laplacian = cv2.convertScaleAbs(laplacian)
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

    io.imshow(laplacian)
    # io.imshow(sobelx)


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

    pic1 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Yao_Ming\\Yao_Ming_0001.jpg")
    pic2 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Amanda_Marsh\\Amanda_Marsh_0001.jpg")
    pic3 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Vincent_Gallo\\Vincent_Gallo_0003.jpg")
    pic4 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\George_HW_Bush\\George_HW_Bush_0008.jpg")
    pic5 = io.imread("D:\\Docs\\Machine Learning\\Data\\LFW\\lfw\\lfw\\Zhang_Ziyi\\Zhang_Ziyi_0003.jpg")

    pic = w*pic1 + w*pic2 + w*pic3 + w*pic4 + w*pic5
    pic = pic.astype(np.uint8)
    io.imshow(pic)
    #
    io.show()
    # cv2.waitKey(0)

experiment()
# generate_mixed_image()
# signal_separation()

