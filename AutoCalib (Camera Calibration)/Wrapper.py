#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2

import math
import argparse
from glob import glob
from scipy import optimize as opt



def displayReprojectedPoints(images_points, optimized_points, images):

    for i, (imagepath, image_points, reprojected_points) in enumerate(zip(images, images_points, optimized_points)):
        image = cv2.imread(imagepath)

        for pt, reprojpt in zip(image_points, reprojected_points):
            x, y = int(pt[0]), int(pt[1])
            x_rp, y_rp = int(reprojpt[0]), int(reprojpt[1])
            cv2.circle(image, (x, y), 15, (0, 255, 0),
                       thickness=5, lineType=8, shift=0)
            cv2.rectangle(image, (x_rp-5, y_rp-5), (
                x_rp+5, y_rp+5), (0, 0, 255), thickness=cv2.FILLED)

        cv2.imwrite("Output/{}.jpg".format(i), image)

def computeExtrinsicParameters(K, lamda, homography_matrix):
    K_inv = np.linalg.inv(K)

    r1 = np.dot(K_inv, homography_matrix[:, 0])
    lamda = np.linalg.norm(r1, ord=2),
    r1 = r1/lamda

    r2 = np.dot(K_inv, homography_matrix[:, 1])
    r2 = r2/lamda

    t = np.dot(K_inv, homography_matrix[:, 2])/lamda

    r3 = np.cross(r1, r2)

    R = np.asarray([r1, r2, r3])
    R = R.T

    return R, t


def nonLinearOptimization(init, lamda, images_points, world_points, homography_matrices):
    K = np.zeros(shape=(3, 3))
    K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1], K[2, 2] = init[0], init[1], init[2], init[3], init[4], 1
    k1, k2 = init[5], init[6]
    u0, v0 = init[2], init[3]

    reprojection_error = np.empty(shape=(1404), dtype=np.float64)
    i = 0
    for image_points, homography_matrix in zip(images_points, homography_matrices):
        R, t = computeExtrinsicParameters(K, lamda, homography_matrix)

        augment = np.zeros((3, 4))
        augment[:, :-1] = R
        augment[:, -1] = t

        for pt, wrldpt in zip(image_points, world_points):
            M = np.array([[wrldpt[0]], [wrldpt[1]], [0], [1]])
            ar = np.dot(augment, M)
            ar = ar/ar[2]
            x, y = ar[0], ar[1]

            U = np.dot(K, ar)
            U = U/U[2]
            u, v = U[0], U[1]

            t = x**2 + y**2
            u_bar = u + (u-u0)*(k1*t + k2*(t**2))
            v_bar = v + (v-v0)*(k1*t + k2*(t**2))

            reprojection_error[i] = pt[0]-u_bar
            i += 1
            reprojection_error[i] = pt[1]-v_bar
            i += 1

    return reprojection_error


def getReprojectionErrorOptimized(image_points, world_points, A, R, t, k1, k2):
    error = 0
    reprojected_points = []

    augment = np.zeros((3, 4))
    augment[:, :-1] = R
    augment[:, -1] = t

    u0, v0 = A[0, 2], A[1, 2]

    for pt, wrldpt in zip(image_points, world_points):
        M = np.array([[wrldpt[0]], [wrldpt[1]], [0], [1]])
        ar = np.dot(augment, M)
        ar = ar/ar[2]
        x, y = ar[0], ar[1]

        U = np.dot(A, ar)
        U = U/U[2]
        u, v = U[0], U[1]

        t = x**2 + y**2
        u_bar = u + (u-u0)*(k1*t + k2*(t**2))
        v_bar = v + (v-v0)*(k1*t + k2*(t**2))

        reprojected_points.append([u_bar, v_bar])

        error = error + np.sqrt((pt[0]-u_bar)**2 + (pt[1]-v_bar)**2)

    return error, reprojected_points


def getReprojectionError(image_points, world_points, A, R, t):
    error = 0
    augment = np.zeros((3, 4))
    augment[:, :-1] = R
    augment[:, -1] = t

    N = np.dot(A, augment)

    for pt, wrldpt in zip(image_points, world_points):
        M = np.array([[wrldpt[0]], [wrldpt[1]], [0], [1]])
        realpt = np.array([[pt[0]], [pt[1]], [1]])
        projpt = np.dot(N, M)
        projpt = projpt/projpt[2]
        diff = realpt - projpt
        error = error + np.linalg.norm(diff, ord=2)

    return error


def getCalibMatrix(b):
    v = (b[0][1]*b[0][3] - b[0][0]*b[0][4])/(b[0][0]*b[0][2] - b[0][1]**2)
    lamda = b[0][5] - (b[0][3]**2 +
                       v*(b[0][1]*b[0][3] - b[0][0]*b[0][4]))/b[0][0]
    alpha = math.sqrt(lamda/b[0][0])
    beta = math.sqrt(lamda*b[0][0]/(b[0][0]*b[0][2] - b[0][1]**2))
    gamma = (-1*b[0][1]*alpha**2*beta)/(lamda)
    u = (gamma*v)/beta - (b[0][3]*alpha**2)/lamda

    print("u = {}\nv = {}\nlamda = {}\nalpha = {}\nbeta = {}\ngamma = {}\n".format(u, v, lamda, alpha, beta, gamma))

    A = np.array([[alpha, gamma, u], [0, beta, v], [0, 0, 1]])
    return A, lamda


def getBMatrix(V):
    _, _, vh = np.linalg.svd(V, full_matrices=True)
    b = vh[-1:]
    return b


def updateVMatrix(H, V):
    v_12 = [H[0][0]*H[0][1], (H[0][0]*H[1][1] + H[1][0]*H[0][1]), H[1][0]*H[1][1],
            (H[2][0]*H[0][1] + H[0][0]*H[2][1]), (H[2][0]*H[1][1] + H[1][0]*H[2][1]), H[2][0]*H[2][1]]

    m1 = H[0][0]*H[0][0] - H[0][1]*H[0][1]
    m2 = 2*(H[0][0]*H[1][0] - H[0][1]*H[1][1])
    m3 = H[1][0]*H[1][0] - H[1][1]*H[1][1]
    m4 = 2*(H[2][0]*H[0][0] - H[0][1]*H[2][1])
    m5 = 2*(H[2][0]*H[1][0] - H[1][1]*H[2][1])
    m6 = H[2][0]*H[2][0] - H[2][1]*H[2][1]

    v_1122 = []
    v_1122.append(m1)
    v_1122.append(m2)
    v_1122.append(m3)
    v_1122.append(m4)
    v_1122.append(m5)
    v_1122.append(m6)

    V.append(v_12)
    V.append(v_1122)

def computeHomography(corners, world_points):
    n = 20
    src = np.asarray(world_points[: n])  # world
    dst = np.asarray(corners[: n])  # image

    P = np.zeros((2*n, 9))

    i = 0
    for (src, dst) in zip(src, dst):
        x, y, x_bar, y_bar = src[0], src[1], dst[0], dst[1]

        P[i][0], P[i][1], P[i][2] = -x, -y, -1
        P[i+1][0], P[i+1][1], P[i+1][2] = 0, 0, 0

        P[i][3], P[i][4], P[i][5] = 0, 0, 0
        P[i+1][3], P[i+1][4], P[i+1][5] = -x, -y, -1

        P[i][6], P[i][7], P[i][8] = x*x_bar, y*x_bar, x_bar
        P[i+1][6], P[i+1][7], P[i+1][8] = x*y_bar, y*y_bar, y_bar

        i = i+2

    _, _, vh = np.linalg.svd(P, full_matrices=True)
    h = vh[-1:]
    h.resize((3, 3))

    homography = h/h[2, 2]
    return homography


def main():
    
    dirpath = 'Data'
    completepath = str(dirpath) + str("/*.jpg")
    images = sorted(glob(completepath))

    V = []

    x, y = np.meshgrid(range(9), range(6))
    world_points = np.hstack((x.reshape(54, 1), y.reshape(54, 1))).astype(np.float32)
    world_points = world_points*21.5
    world_points = np.asarray(world_points)

    images_points = []
    homography_matrices = []

    for imagepath in images:
        image = cv2.imread(imagepath)
        scale_percent = 30  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, dim)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            corners = corners.reshape(-1, 2)
            cv2.drawChessboardCorners(gray, (9, 6), corners, ret)

            homography_matrix = computeHomography(corners, world_points)
            images_points.append(corners)
            homography_matrices.append(homography_matrix)
            updateVMatrix(homography_matrix, V)

        cv2.destroyAllWindows()

    V = np.asarray(V)
    b = getBMatrix(V)

    K, lamda = getCalibMatrix(b)
    print("Initial estimate of Calibration matrix: \n\n{}".format(K))

    R, t = computeExtrinsicParameters(K, lamda, homography_matrix)
    print("Initial estimate of Extrinsic parameters: \nRotation Matrix: \n\n {} \n\nTransaltion Vector: \n\n {}".format(R, t))

    error = 0
    for image_points, homography_matrix in zip(images_points, homography_matrices):
        R, t = computeExtrinsicParameters(K, lamda, homography_matrix)

        reprojection_error = getReprojectionError(
            image_points, world_points, K, R, t)

        error = error + reprojection_error

    error = error/(13*9*6)
    print("\nMean Reprojection error before optimization: \n{}".format(error))

    init = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1], 0, 0]
    res = opt.least_squares(fun=nonLinearOptimization, x0=init, method="lm", args=[lamda, images_points, world_points, homography_matrices])

    K_opt = np.zeros(shape=(3, 3))
    K_opt[0, 0], K_opt[1, 1], K_opt[0, 2], K_opt[1, 2], K_opt[0, 1], K_opt[2, 2] = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 1

    k1_opt, k2_opt = res.x[5], res.x[6]

    print("\nCalibration matrix after optimization: \n\n{}".format(K_opt))
    print("\nDistortion coefficients after optimization: \n{}, {}".format(k1_opt, k2_opt))

    error_opt = 0
    optimized_points = []
    for image_points, homography_matrix in zip(images_points, homography_matrices):
        R, t = computeExtrinsicParameters(K_opt, lamda, homography_matrix)

        reprojection_error, reprojected_points = getReprojectionErrorOptimized(
            image_points, world_points, K_opt, R, t, k1_opt, k2_opt)
        optimized_points.append(reprojected_points)

        error_opt = error_opt + reprojection_error

    error_opt = error_opt/(13*9*6)
    print("\nMean Reprojection error after optimization: \n{}".format(
        error_opt[0]))

    displayReprojectedPoints(images_points, optimized_points, images)


if __name__ == '__main__':
    main()



