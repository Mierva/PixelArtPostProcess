import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class ContoursDetector:
    def __init__(self, image_path):
        self.input_image_path = image_path
        if os.path.exists(image_path):
            self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            raise Exception('Image not found')
        
    def find_general_contour(self, kernel_size_to_dilate, kernel_size_to_smooth):
        kernel_to_dilate = np.ones((kernel_size_to_dilate, kernel_size_to_dilate), np.uint8)
        
        if self.image.shape == 4:
            dilate_image = cv2.dilate(self.image, kernel_to_dilate, iterations=1)
        else:    
            raise Exception('Wrong image shape, image shape must be 4 channel')
        
        
        smoothed_image = cv2.medianBlur(dilate_image, kernel_size_to_smooth) 
        bgr = smoothed_image[:, :, :3]
        alpha_channel = smoothed_image[:, :, 3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_with_alpha = cv2.merge((gray, gray, gray, alpha_channel))

        image_with_contour = self.image.copy()

        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_with_alpha[:, :, 0])
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        image_with_contour[mask == 0] = [0, 0, 0, 0]
        color = [157, 133, 52, 255]

        cv2.drawContours(image_with_contour, contours, -1, color, thickness=1)

        cv2.imwrite("/home/nikolay/aseprite/image_data/whale_second_process/image_with_contour_car.png", image_with_contour)
        return image_with_contour
    
if __name__ == '__main__':
    image_path = '/home/nikolay/aseprite/image_data/presentation/try_2.png'
    bebra = ContoursDetector(image_path)
    bebra.find_general_contour(5, 5)