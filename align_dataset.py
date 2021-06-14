import os
import tensorflow as tf
from align.detect_face import create_mtcnn, detect_face
from align.mtcnn_detector import extract_image_chips
import numpy as np
from imageio import imread, imwrite
from skimage import transform, img_as_ubyte
from tqdm import tqdm
import argparse
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    input_dir = args.input_dir
    output_dir =args.output_dir
    image_size = args.image_size
    padding = 0.3

    # MTCNN parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps' threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        with sess.as_default():
            pnet, rnet, onet = create_mtcnn(sess, model_path='align/model')

    file_paths = [os.path.join(root,file) for root, directory, files in os.walk(input_dir) for file in files
                 if file.endswith('.jpg') or file.endswith('.png')]
    nrof_successfully_aligned = 0
    for file_path in tqdm(file_paths, desc='Aligning... ', unit='imgs'):
        output_file_path = file_path.replace(input_dir, output_dir).replace('.jpg', '.png')
        output_sub_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir, exist_ok=True)

        if not os.path.exists(output_file_path):
            try:
                img = imread(file_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(file_path, e)
                print(errorMessage)
            else:
                if len(img.shape) == 2:
                    img = np.tile(np.expand_dims(img, -1), [1, 1, 3])
                img = img[:, :, 0:3]

                bbxs, lms = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                if bbxs.shape[0] > 0:
                    img_size = np.asarray(img.shape)[0:2]
                    if bbxs.shape[0] > 1:
                        # Handle mulitple faces
                        bounding_box_size = (bbxs[:, 2] - bbxs[:, 0]) * (bbxs[:, 3] - bbxs[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack([(bbxs[:, 0] + bbxs[:, 2]) / 2 - img_center[1],
                                             (bbxs[:, 1] + bbxs[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                    else:
                        index = 0

                    aligned = extract_image_chips(img=img, points=np.transpose(np.expand_dims(lms[:, index], 1)),
                                                  desired_size=image_size, padding=padding)[0]
                    nrof_successfully_aligned += 1
                else:
                    aligned = transform.resize(img, (image_size, image_size), order=1)
                imwrite(output_file_path, img_as_ubyte(aligned))

    print('\nTotal number of images: %d' % len(file_paths))
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='Path to the dataset which will be aligned.',
                        default='/mnt/ssd/datasets/LFW/LFW')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the directy for the aligned faces.',
                        default='/mnt/ssd/datasets/LFW/LFW_aligned')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
