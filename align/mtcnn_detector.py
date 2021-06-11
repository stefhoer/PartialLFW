# coding: utf-8
# https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection/blob/master/mtcnn_detector.py
import numpy as np
import math
import cv2

izip = zip


def list2colmatrix(pts_list):
    """
        convert list to column matrix
    Parameters:
    ----------
        pts_list:
            input list
    Retures:
    -------
        colMat:

    """
    assert len(pts_list) > 0
    colMat = []
    for i in range(len(pts_list)):
        colMat.append(pts_list[i][0])
        colMat.append(pts_list[i][1])
    colMat = np.matrix(colMat).transpose()
    return colMat


def find_tfrom_between_shapes(from_shape, to_shape):
    """
        find transform between shapes
    Parameters:
    ----------
        from_shape:
        to_shape:
    Retures:
    -------
        tran_m:
        tran_b:
    """
    assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

    sigma_from = 0.0
    sigma_to = 0.0
    cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

    # compute the mean and cov
    from_shape_points = from_shape.reshape(round(from_shape.shape[0] / 2), 2)
    to_shape_points = to_shape.reshape(round(to_shape.shape[0] / 2), 2)
    mean_from = from_shape_points.mean(axis=0)
    mean_to = to_shape_points.mean(axis=0)

    for i in range(from_shape_points.shape[0]):
        temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
        sigma_from += temp_dis * temp_dis
        temp_dis = np.linalg.norm(to_shape_points[i] - mean_to)
        sigma_to += temp_dis * temp_dis
        cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

    sigma_from = sigma_from / to_shape_points.shape[0]
    sigma_to = sigma_to / to_shape_points.shape[0]
    cov = cov / to_shape_points.shape[0]

    # compute the affine matrix
    s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    u, d, vt = np.linalg.svd(cov)

    if np.linalg.det(cov) < 0:
        if d[1] < d[0]:
            s[1, 1] = -1
        else:
            s[0, 0] = -1
    r = u * s * vt
    c = 1.0
    if sigma_from != 0:
        c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

    tran_b = mean_to.transpose() - c * r * mean_from.transpose()
    tran_m = c * r

    return tran_m, tran_b


def extract_image_chips(img, points, desired_size=256, padding=0):
    """
        crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
        crop_imgs: list, n
            cropped and aligned faces
    """
    crop_imgs = []
    for p in points:
        shape = []
        for k in range(round(len(p) / 2)):
            if p[k] is not None:
                shape.append(p[k])
                shape.append(p[k + 5])

        if padding > 0:
            padding = padding
        else:
            padding = 0

        # average positions of face points
        mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
        mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

        mean_face_shape_x = [0.2194, 0.7747, 0.4971, 0.3207, 0.6735]
        mean_face_shape_y = [0.1871, 0.1871, 0.5337, 0.7633, 0.7633]

        from_points = []
        to_points = []

        for i in range(round(len(shape) / 2)):
            x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
            y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
            to_points.append([x, y])
            from_points.append([shape[2 * i], shape[2 * i + 1]])

        # convert the points to Mat
        from_mat = list2colmatrix(from_points)
        to_mat = list2colmatrix(to_points)

        # compute the similar transfrom
        tran_m, tran_b = find_tfrom_between_shapes(from_mat, to_mat)

        probe_vec = np.matrix([1.0, 0.0]).transpose()
        probe_vec = tran_m * probe_vec

        scal = np.linalg.norm(probe_vec)
        angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

        from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
        to_center = [0, 0]
        to_center[1] = desired_size * 0.4
        to_center[0] = desired_size * 0.5

        ex = to_center[0] - from_center[0]
        ey = to_center[1] - from_center[1]

        rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scal)
        rot_mat[0][2] += ex
        rot_mat[1][2] += ey

        chips = cv2.warpAffine(img, rot_mat, (desired_size, desired_size), borderValue=(128, 128, 128))
        crop_imgs.append(chips)

    return crop_imgs

