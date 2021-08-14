import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


class edge_detection:

    def gaussianMask(self, Sizeofthemask, sigma=1):

        kernel_1D = np.linspace(-(Sizeofthemask // 2), Sizeofthemask // 2, Sizeofthemask)
        for i in range(Sizeofthemask):
            kernel_1D[i] = 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power((kernel_1D[i]) / sigma, 2) / 2)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

        plt.figure(2)
        plt.imshow(kernel_2D, cmap='gray')
        plt.title("gaussian mask")
        plt.xticks([]), plt.yticks([])

        return kernel_2D

    def convolute(self, image, filter):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        image_row, image_col = image.shape
        kernel_row, kernel_col = filter.shape
        output = np.zeros(image.shape)
        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)
        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(filter * padded_image[row:row + kernel_row, col:col + kernel_col])

        return output

    def firstDerivativeEdgeDetector(self, image):

        mask = np.array([[-1, 0, 1]])
        kernel = self.gaussianMask(3, sigma=1)
        gau_smoothing =  self.convolute(image, kernel)

        plt.figure(3)
        plt.imshow(gau_smoothing, cmap='gray')
        plt.title("image after passing gaussian filter")
        plt.xticks([]), plt.yticks([])

        new_image_x = self.convolute(gau_smoothing, mask)
        new_image_y = self.convolute(gau_smoothing, np.flip(mask.T, axis=1))

        plt.figure(4)
        plt.imshow(new_image_x, cmap='gray')
        plt.title("x-direction (first derivative)")
        plt.xticks([]), plt.yticks([])

        plt.figure(5)
        plt.imshow(new_image_y, cmap='gray')
        plt.title("y-direction (first derivative)")
        plt.xticks([]), plt.yticks([])

        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

        plt.figure(6)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("gradient magnitude")
        plt.xticks([]), plt.yticks([])

        gradient_direction = np.arctan2(new_image_y, new_image_x)
        gradient_direction = np.rad2deg(gradient_direction)

        return gradient_magnitude , gradient_direction

    def secondDerivativeEdgeDetector(self, image):

        mask = np.array([[1, -2, 1]])
        kernel = self.gaussianMask(3, sigma=1)
        gau_smoothing = self.convolute(image, kernel)

        plt.figure(3)
        plt.imshow(gau_smoothing, cmap='gray')
        plt.title("image after passing gaussian filter")
        plt.xticks([]), plt.yticks([])

        new_image_x = self.convolute(gau_smoothing, mask)
        new_image_y = self.convolute(gau_smoothing, np.flip(mask.T, axis=1))

        plt.figure(4)
        plt.imshow(new_image_x, cmap='gray')
        plt.title("x-direction (second derivative)")
        plt.xticks([]), plt.yticks([])

        plt.figure(5)
        plt.imshow(new_image_y, cmap='gray')
        plt.title("y-direction (second derivative)")
        plt.xticks([]), plt.yticks([])

        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

        plt.figure(6)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("gradient magnitude")
        plt.xticks([]), plt.yticks([])

        gradient_direction = np.arctan2(new_image_y, new_image_x)
        gradient_direction = np.rad2deg(gradient_direction)

        return gradient_magnitude, gradient_direction

    def sobil(self, image):

        mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
        kernel = self.gaussianMask(3, sigma=1)
        gau_smoothing = self.convolute(image, kernel)

        plt.figure(3)
        plt.imshow(gau_smoothing, cmap='gray')
        plt.title("image after passing gaussian filter")
        plt.xticks([]), plt.yticks([])

        new_image_x = self.convolute(gau_smoothing, mask)
        new_image_y = self.convolute(gau_smoothing, np.flip(mask.T, axis=1))

        plt.figure(4)
        plt.imshow(new_image_x, cmap='gray')
        plt.title("x-direction (sobil)")
        plt.xticks([]), plt.yticks([])

        plt.figure(5)
        plt.imshow(new_image_y, cmap='gray')
        plt.title("y-direction (sobil)")
        plt.xticks([]), plt.yticks([])

        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

        plt.figure(6)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("gradient magnitude")
        plt.xticks([]), plt.yticks([])

        gradient_direction = np.arctan2(new_image_y, new_image_x)
        gradient_direction = np.rad2deg(gradient_direction)

        return gradient_magnitude, gradient_direction

    def prewitt(self, image):

        mask = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
        kernel = self.gaussianMask(3, sigma=1)
        gau_smoothing = self.convolute(image, kernel)

        plt.figure(3)
        plt.imshow(gau_smoothing, cmap='gray')
        plt.title("image after passing gaussian filter")
        plt.xticks([]), plt.yticks([])

        new_image_x = self.convolute(gau_smoothing, mask)
        new_image_y = self.convolute(gau_smoothing, np.flip(mask.T, axis=1))

        plt.figure(4)
        plt.imshow(new_image_x, cmap='gray')
        plt.title("x-direction (prewitt)")
        plt.xticks([]), plt.yticks([])

        plt.figure(5)
        plt.imshow(new_image_y, cmap='gray')
        plt.title("y-direction (prewitt)")
        plt.xticks([]), plt.yticks([])

        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

        plt.figure(6)
        plt.imshow(gradient_magnitude, cmap='gray')
        plt.title("gradient magnitude")
        plt.xticks([]), plt.yticks([])

        gradient_direction = np.arctan2(new_image_y, new_image_x)
        gradient_direction = np.rad2deg(gradient_direction)

        return gradient_magnitude, gradient_direction

    def non_maxima_suppression(self, image, angels):   # input (gradient magnitude and gradient direction)

        image_row, image_col = image.shape
        output = np.zeros(image.shape)

        PI = 180
        for row in range(1, image_row - 1):
            for col in range(1, image_col - 1):
                direction = angels[row, col]

                if direction < 0:
                    direction = direction + 360

                if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI) or (7 * PI / 8 <= direction < 9 * PI / 8):
                    before_pixel = image[row, col - 1]
                    after_pixel = image[row, col + 1]

                elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                    before_pixel = image[row + 1, col - 1]
                    after_pixel = image[row - 1, col + 1]

                elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                    before_pixel = image[row - 1, col]
                    after_pixel = image[row + 1, col]

                else:
                    before_pixel = image[row - 1, col - 1]
                    after_pixel = image[row + 1, col + 1]

                if image[row, col] >= before_pixel and image[row, col] >= after_pixel:
                    output[row, col] = image[row, col]

        plt.figure(7)
        plt.imshow(output, cmap='gray')
        plt.title("non max suppression")
        plt.xticks([]), plt.yticks([])

        return output

    def double_threshold(self, image):

        output = np.zeros(image.shape)
        high = 60
        low = 20
        strong = 255
        weak = 50

        strong_row, strong_col = np.where(image > high)
        weak_row, weak_col = np.where((image <= high) & (image >= low))

        output[strong_row, strong_col] = strong
        output[weak_row, weak_col] = weak

        plt.figure(8)
        plt.imshow(output, cmap='gray')
        plt.title("threshold")
        plt.xticks([]), plt.yticks([])

        return output

    def edge_linking(self, image):
        weak = 50
        image_row, image_col = image.shape

        n_img = image.copy()

        for row in range(0, image_row):
            for col in range(0, image_col):
                if n_img[row, col] == weak:
                    if n_img[row, col + 1] == 255 or n_img[row, col - 1] == 255 or \
                            n_img[row - 1, col] == 255 or n_img[row + 1, col] == 255 or \
                            n_img[row - 1, col - 1] == 255 or n_img[row + 1, col - 1] == 255 or \
                            n_img[row - 1, col + 1] == 255 or n_img[row + 1, col + 1] == 255:
                        n_img[row, col] = 255
                    else:
                        n_img[row, col] = 0

        plt.figure(9)
        plt.imshow(n_img, cmap='gray')
        plt.title("Canny Edge Detector")
        plt.xticks([]), plt.yticks([])

        return n_img

    def canny(self, image):

        gradient_magnitude, gradient_direction = self.sobil(image)
        new_image = self.non_maxima_suppression(gradient_magnitude, gradient_direction)
        new_image = self.double_threshold(new_image)
        new_image = self.edge_linking(new_image)
        return new_image

def main():
    edge = edge_detection()

    img = cv2.imread('photo.png')

    plt.figure(1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.xticks([]), plt.yticks([])

    #edge.firstDerivativeEdgeDetector(img)
    #edge.secondDerivativeEdgeDetector(img)
    #edge.sobil(img)
    #edge.prewitt(img)
    edge.canny(img)

    plt.show()


if __name__ == "__main__":
    main()
