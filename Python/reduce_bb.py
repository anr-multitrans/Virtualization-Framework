import cv2
import numpy as np
object_color = (142, 0, 0)
# Read the semantic segmentation image from the file system
sem_seg_image = cv2.imread('image_semseg.png', cv2.IMREAD_UNCHANGED)

sem_seg_image = sem_seg_image[..., :3]
# Instantiate the min/max x/y coordinates
x_min, x_max, y_min, y_max = 485, 550, 510, 534

#center_pixel = sem_seg_image[503, 520]
#print(center_pixel)
#color = np.array([center_pixel[2], center_pixel[1], center_pixel[0]])
#print(color)

# Extract the object color from the semantic segmentation image at the center pixel
#ocenter_pixel = sem_seg_image[sem_seg_image.shape[0]//2, sem_seg_image.shape[1]//2]


# Extract the object mask from the segmentation image
object_mask = cv2.inRange(sem_seg_image, object_color, object_color)

# Save the object mask as an image
cv2.imwrite("object_mask.png", object_mask)

# Crop the object mask to the bounding box
object_mask_bb = object_mask[y_min:y_max+1, x_min:x_max+1]

# Save the cropped object mask as an image
cv2.imwrite("object_mask_bb.png", object_mask_bb)

# Find the indices of all pixels that belong to the object
object_pixels_indices = np.argwhere(object_mask_bb != 0)

x_min_new = x_min + np.min(object_pixels_indices[:, 1])
y_min_new = y_min + np.min(object_pixels_indices[:, 0])
x_max_new = x_min + np.max(object_pixels_indices[:, 1])
y_max_new = y_min + np.max(object_pixels_indices[:, 0])
