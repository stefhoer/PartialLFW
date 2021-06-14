import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


def main(args):
    input_dir = args.input_dir
    output_dir =args.output_dir
    image_size = args.image_size
    PADDING = 0.3

    FACTORS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    PARTS = ['lEye', 'rEye', 'Nose', 'Mouth']

    mean_face_shape_x = [0.2194, 0.7747, 0.4971, 0.3207, 0.6735]
    mean_face_shape_y = [0.1871, 0.1871, 0.5337, 0.7633, 0.7633]
    # Take the mean x-coordinate for mouth
    mean_face_shape_x[3] = 0.5 * (mean_face_shape_x[3] + mean_face_shape_x[4])

    # get 4 landmark coordinates
    point_dict = {PARTS[idx]: [(PADDING + mean_face_shape_x[idx]) / (2 * PADDING + 1) * image_size,
                               (PADDING + mean_face_shape_y[idx]) / (2 * PADDING + 1) * image_size + 15]
                  for idx in range(4)}

    file_paths = [os.path.join(root,file) for root, directory, files in os.walk(input_dir) for file in files
                 if file.endswith('.jpg') or file.endswith('.png')]

    for file_path in tqdm(file_paths, desc='Generate partial faces... ', unit='imgs'):
        try:
            img = cv2.imread(file_path)
        except (IOError, ValueError, IndexError) as e:
            print('{}: {}'.format(file_path, e))
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
                    output_file_path = file_path.replace(input_dir, os.path.join(output_dir, part, 'factor_' + str(factor))).replace('.jpg', '.png')
                    if not os.path.exists(output_file_path):
                        if not os.path.exists(os.path.dirname(output_file_path)):
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        cv2.imwrite(output_file_path, partial)

                    # Center partial image
                    dx = 80 - point_dict[part][0]
                    dy = 80 - point_dict[part][1]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    partial_centered = cv2.warpAffine(partial, M, (image_size, image_size), borderValue=(128, 128, 128))

                    output_file_path = file_path.replace(input_dir, os.path.join(output_dir, part + '_centered', 'factor_' + str(factor))).replace('.jpg', '.png')
                    # Save
                    if not os.path.exists(output_file_path):
                        if not os.path.exists(os.path.dirname(output_file_path)):
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        cv2.imwrite(output_file_path, partial_centered)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='Path to the dataset which will be aligned.',
                        default='/mnt/ssd/datasets/LFW/LFW_aligned')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the directy for the aligned faces.',
                        default='/mnt/ssd/datasets/LFW/PartialLFW')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    main(parser.parse_args(sys.argv[1:]))
