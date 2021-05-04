import cv2
import numpy as np

if __name__ == "__main__":
    image = cv2.imread("resources/chessboard.jpg")
    image_grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(image_grayed, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image_grayed, cv2.CV_32F, 0, 1, ksize=1)

    ixx = gx*gx
    iyy = gy*gy
    ixy = gx*gy

    window = 3
    window_step = window//2
    k = 0.04
    rows, cols = image_grayed.shape

    # Concise way
    sxx = cv2.filter2D(ixx, -1, np.ones((window, window)))
    syy = cv2.filter2D(iyy, -1, np.ones((window, window)))
    sxy = cv2.filter2D(ixy, -1, np.ones((window, window)))

    det = (sxx * syy) - (sxy ** 2)
    trace = sxx + syy
    r = det - k * (trace ** 2)

    threshold = 0.01 * r.max()
    image[r > threshold] = (0, 0, 255)

    # Detailed way
    # r = np.zeros((rows, cols))
    # for row in range(window_step, rows-window_step):
    #     for col in range(window_step, cols-window_step):
    #         sxx = ixx[row-window_step: row+window_step, col-window_step: col+window_step].sum()
    #         syy = iyy[row-window_step: row+window_step, col-window_step: col+window_step].sum()
    #         sxy = ixy[row-window_step: row+window_step, col-window_step: col+window_step].sum()
    #
    #         m = np.array([[sxx, sxy], [sxy, syy]])
    #         r.itemset((row, col), np.linalg.det(m) - k * ((m.trace()) ** 2))
    #
    # image[r > 0.1*r.max()] = (0, 0, 255)

    cv2.imshow("Harris", image)
    cv2.waitKey()
