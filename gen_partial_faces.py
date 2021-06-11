import cv2
import os
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = 160
PADDING = 0.3
base_dir = '/mnt/ssd/datasets/LFW'
INPUT_DIR = os.path.join(base_dir, 'LFW_aligned')
OUTPUT_BASE_DIR = os.path.join(base_dir, 'PartialLFW')

FACTORS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
PARTS = ['lEye', 'rEye', 'Nose', 'Mouth']

mean_face_shape_x = [0.2194, 0.7747, 0.4971, 0.3207, 0.6735]
mean_face_shape_y = [0.1871, 0.1871, 0.5337, 0.7633, 0.7633]
# Take the mean x-coordinate for mouth
mean_face_shape_x[3] = 0.5 * (mean_face_shape_x[3] + mean_face_shape_x[4])

# get 4 landmark coordinates
point_dict = {PARTS[idx]: [(PADDING + mean_face_shape_x[idx]) / (2 * PADDING + 1) * IMAGE_SIZE,
                           (PADDING + mean_face_shape_y[idx]) / (2 * PADDING + 1) * IMAGE_SIZE + 15]
              for idx in range(4)}


nrof_images_total = sum([len(file) for root, _, file in os.walk(INPUT_DIR)])

with tqdm(total=nrof_images_total, desc='Generate partial faces... ', unit='imgs') as pbar:
    for root, directory, filenames in os.walk(INPUT_DIR):
        if filenames:
            class_dir = root[len(INPUT_DIR) + 1:]
            for part in PARTS:
                for factor in FACTORS:
                    output_class_dir = os.path.join(OUTPUT_BASE_DIR, part, 'factor_' + str(factor), class_dir)
                    if not os.path.exists(output_class_dir):
                        os.makedirs(output_class_dir, exist_ok=True)
                    output_class_dir = os.path.join(OUTPUT_BASE_DIR, part + '_centered', 'factor_' + str(factor), class_dir)
                    if not os.path.exists(output_class_dir):
                        os.makedirs(output_class_dir, exist_ok=True)

            for filename in [file for file in filenames if file.endswith('.png')]:
                file_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(file_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(file_path, e)
                    print(errorMessage)
                else:
                    img = img[:, :, 0:3] - 128
                    for part in PARTS:
                        for factor in FACTORS:
                            # Create Mask
                            roi = np.zeros(img.shape[0:2], dtype=np.uint8)
                            x1 = int(point_dict[part][0] - (15 + 120 * factor))
                            y1 = int(point_dict[part][1] - (10 + 105 * factor))
                            x2 = int(point_dict[part][0] + (15 + 120 * factor))
                            y2 = int(point_dict[part][1] + (10 + 105 * factor))
                            cv2.rectangle(roi, (x1, y1), (x2, y2), 1, -1)

                            partial = cv2.bitwise_and(img, img, mask=roi) + 128

                            # Save
                            save_path = os.path.join(OUTPUT_BASE_DIR, part, 'factor' + '_' + str(factor),
                                                     class_dir, filename)
                            if not os.path.exists(save_path):
                                cv2.imwrite(save_path, partial)

                            # Center partial image
                            dx = 80 - point_dict[part][0]
                            dy = 80 - point_dict[part][1]
                            M = np.float32([[1, 0, dx], [0, 1, dy]])
                            partial_centered = cv2.warpAffine(partial, M, (IMAGE_SIZE, IMAGE_SIZE), borderValue=(128, 128, 128))

                            save_path = os.path.join(OUTPUT_BASE_DIR, part + '_centered', 'factor' + '_' + str(factor),
                                                     class_dir, filename)
                            # Save
                            if not os.path.exists(save_path):
                                cv2.imwrite(save_path, partial_centered)
                pbar.update(1)