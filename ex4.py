import numpy as np
import cv2
background_img_1 = cv2.imread('GreenBackground.png', 1)
object_img = cv2.imread('Object.png', 1)
background_img_2 = cv2.imread('NewBackground.png', 1)
img_size = (678, 318)
background_img_1_resize = cv2.resize(background_img_1, img_size)
object_img_resize = cv2.resize(object_img, img_size)
background_img_2_resize = cv2.resize(background_img_2, img_size)
differences = cv2.absdiff(object_img_resize, background_img_1_resize)
difference_single = np.sum(differences, axis = 2) / 3
difference_single = difference_single.astype(np.unit8)
def compute_binary_mask(different_single_channel):
    difference_binary = np.where(different_single_channel >=15, 255, 0)
    difference_binary = np.stack((difference_binary,)*3, axis = -1)
    return difference_binary
binary_mask = compute_binary_mask(difference_single)
output = np.where(binary_mask == 255, object_img_resize, background_img_2_resize)
