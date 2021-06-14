import os
import tensorflow as tf
from align.detect_face import create_mtcnn, detect_face
from align.mtcnn_detector import extract_image_chips
import numpy as np
from imageio import imread, imwrite
from skimage import transform, img_as_ubyte
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_SIZE = 160
PADDING = 0.3

base_dir = '/mnt/ssd/datasets/LFW/'
INPUT_DIR = os.path.join(base_dir, 'LFW')
OUTPUT_DIR = os.path.join(base_dir, 'LFW_aligned')
MODEL_DIR = 'align/model'

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    with sess.as_default():
        pnet, rnet, onet = create_mtcnn(sess, MODEL_DIR)

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps' threshold
factor = 0.709  # scale factor

nrof_images_total = sum([len(file) for root, _, file in os.walk(INPUT_DIR)])

with tqdm(total=nrof_images_total, desc='Aligning... ', unit='imgs') as pbar:
    nrof_successfully_aligned = 0
    for root, directory, filenames in os.walk(INPUT_DIR):
        if filenames:
            class_dir = root[len(INPUT_DIR) + 1:]
            output_class_dir = os.path.join(OUTPUT_DIR, class_dir)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir, exist_ok=True)
            for filename in [file for file in filenames if file.endswith('.jpg') or file.endswith('.png')]:
                file_path = os.path.join(root, filename)
                output_filename = os.path.join(OUTPUT_DIR, class_dir,
                                               '.'.join(filename.split('.')[:-1]) + '.png')
                if not os.path.exists(output_filename):
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
                                                          desired_size=IMAGE_SIZE, padding=PADDING)[0]
                            nrof_successfully_aligned += 1
                        else:
                            aligned = transform.resize(img, (IMAGE_SIZE, IMAGE_SIZE), order=1)
                        imwrite(output_filename, img_as_ubyte(aligned))

                pbar.update(1)

print('\nTotal number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
