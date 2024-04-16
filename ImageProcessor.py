from PIL import Image, ImageFilter
import numpy as np
import cairosvg
from scipy import stats
import cv2


class ImageProcessor:
    def __init__(self, image_path, threshold=150, neighbors_number=4):
        self.image_path = image_path
        self.threshold = threshold
        self.neighbors_number = neighbors_number
        self.output_png_image_path = image_path.replace('.svg', '.png')
        self.gray_without_alpha_to_CNN = self.output_png_image_path.replace('.png', '_gray_without_alpha_to_CNN.png')
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.update_processing_parameters(threshold, neighbors_number)
        
    def update_processing_parameters(self, threshold, neighbors_number):
        self.threshold = threshold
        self.neighbors_number = neighbors_number

    @staticmethod
    def most_frequent_color(window):
        unique, counts = np.unique(window, return_counts=True)
        most_frequent_index = np.argmax(counts)
        return unique[most_frequent_index]

    @staticmethod
    def compare_pixel_with_neighbors(pixel, neighbors, threshold, neighbors_number):
        count = 0
        for neighbor in neighbors:
            if np.abs(int(pixel) - int(neighbor)) >= threshold:
                count += 1
        return count >= neighbors_number
    
    # @staticmethod
    # def compare_pixel_with_neighbors(pixel, neighbors, threshold, neighbors_number):
    #     count = 0

    #     # Convert pixel to integer and check if conversion is successful
    #     try:
    #         pixel_value = int(pixel)
    #     except ValueError:
    #         print(f"Conversion error: Pixel value '{pixel}' is not an integer.")
    #         return False  # Return false or handle in another appropriate way

    #     for neighbor in neighbors:
    #         # Convert neighbor to integer and check if conversion is successful
    #         try:
    #             neighbor_value = int(neighbor)
    #         except ValueError:
    #             print(f"Conversion error: Neighbor value '{neighbor}' is not an integer.")
    #             continue  # Optionally continue to next neighbor

    #         # Perform the comparison with absolute values
    #         try:
    #             if np.abs(pixel_value - neighbor_value) >= threshold:
    #                 count += 1
    #         except TypeError as e:
    #             print(f"TypeError in comparison: {e}")
    #             continue

    #     return count >= neighbors_number


    # @staticmethod
    # def iterate_neighbors(img, x, y):
    #     neighbors = []
    #     for dy in range(-1, 2):
    #         for dx in range(-1, 2):
    #             if dx == 0 and dy == 0:
    #                 continue
    #             nx, ny = x + dx, y + dy
    #             if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
    #                 neighbors.append(img[ny, nx])
    #     print(f"Neighbors: {neighbors}")
    #     return neighbors

    def iterate_neighbors(self, img, x, y):
        neighbors = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0]:
                    neighbors.append(img[ny, nx])
        
        # Фильтрация ненулевых соседей
        nonzero_neighbors = [neighbor for neighbor in neighbors if not np.all(neighbor == 0)]
        
        if nonzero_neighbors:
            print(f"Non-zero Neighbors: {nonzero_neighbors}")
        else:
            print("No non-zero neighbors found.")
        
        return nonzero_neighbors

    
    def four_channel_png_to_bgr(self):
        img = cv2.imread(self.output_png_image_path, cv2.IMREAD_UNCHANGED)
        bgr = img[:, :, 0:3]
        alpha_channel = img[:, :, 3]
        gray_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gray_without_alpha = cv2.merge((gray_image, gray_image, gray_image))
        # cv2.imwrite(self.gray_without_alpha_to_CNN, gray_without_alpha, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return alpha_channel
    
    def from_svg_to_png(self):
        cairosvg.svg2png(url=self.image_path, write_to=self.output_png_image_path)
        image = Image.open(self.output_png_image_path)

        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        image_np = np.array(image)        
        cv2.imwrite(self.output_png_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA))

        # Update self.image_path to point to the newly created PNG image
        self.image_path = self.output_png_image_path
        
        # Update self.image to ensure the changes are applied to the newly created PNG image
        self.image = cv2.imread(self.output_png_image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Unable to load image from {self.output_png_image_path}")

    def change_isolating_pixels(self):
        if self.image is None:
            raise ValueError(f"Failed to load image from path: {self.image_path}")

        if self.image.shape[2] == 4:
            img = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha_channel = self.image[:, :, 3]
        else:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            alpha_channel = None

        modified_image = img.copy()
        changes = []

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                pixel = img[y, x]
                neighbors = self.iterate_neighbors(img, x, y)
                if self.compare_pixel_with_neighbors(pixel, neighbors, self.threshold, self.neighbors_number):
                    most_common_color = self.most_frequent_color(np.array(neighbors))
                    changes.append(((x, y), pixel, most_common_color))
                    modified_image[y, x] = most_common_color

        if alpha_channel is not None:
            modified_image_with_alpha = cv2.merge((modified_image, modified_image, modified_image, alpha_channel))
        else:
            modified_image_with_alpha = cv2.merge((modified_image, modified_image, modified_image))

        return modified_image_with_alpha, changes

    def save_modified_image(self, output_path):
        modified_image, _ = self.change_isolating_pixels()
        cv2.imwrite(output_path, modified_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return modified_image, output_path
    
    def convert_from_3_to_4_channel_image(self, result_image, alpha_channel):
        if isinstance(result_image, Image.Image):
            result_image = np.array(result_image)
            
        if alpha_channel.ndim == 2:  
            alpha_channel = alpha_channel.reshape(alpha_channel.shape[0], alpha_channel.shape[1], 1)
            
        # print("result_image shape:", result_image.shape)
        # print("alpha_channel shape:", alpha_channel.shape)
        # print("result_image dtype:", result_image.dtype)
        # print("alpha_channel dtype:", alpha_channel.dtype)

        alpha_channel_resized = cv2.resize(alpha_channel, (result_image.shape[1], result_image.shape[0]))
        full_img = cv2.merge((result_image[:, :, 0], result_image[:, :, 1], result_image[:, :, 2], alpha_channel_resized))

        
        return full_img

    
    def find_main_colors_in_contour(self, image, contour, num_colors):
        mask = np.zeros_like(image[:, :, 0])
        cv2.drawContours(mask, [contour], 0, (255), thickness=cv2.FILLED)
    
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGRA2RGB)
        
        pixels = np.float32(rgb.reshape(-1, 3))
        
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        most_popular_color_indexes = np.argsort(counts)[-num_colors:]
        
        main_colors = unique_colors[most_popular_color_indexes]
        
        return main_colors, counts
    
    def find_most_sutable_color(self, four_channel_image, contour, kenel_size = 2, num_colors = 7):
        kernel = np.ones((kenel_size, kenel_size), np.uint8)
        dilate_img = cv2.dilate(four_channel_image, kernel, iterations=1)
        bgr = dilate_img[:, :, 0:3]  
        alpha_channel = dilate_img[:, :, 3]
        img_gray = cv2.cvtColor(dilate_img, cv2.COLOR_BGR2GRAY)
        gray_with_alpha = cv2.merge((img_gray, img_gray, img_gray, alpha_channel))
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        main_colors_per_contour = []
        for contour in contours:
            main_colors, number_of_colors = self.find_main_colors_in_contour(four_channel_image, contour, num_colors)
            for color in main_colors:
                if color[0] != 0 and color[1] != 0 and color[2] != 0:
                    # print(f"Main color:{main_colors}")
                    main_colors_per_contour.append(main_colors)
        # for i, main_colors in enumerate(main_colors_per_contour):
        #     print(f"Main colors for contour {i + 1}:")
        #     for color in main_colors:
        #         print(color)

        return main_colors
    
    def reshape_image(self, image_shape):
        temp = image_shape[0]
        image_shape[0] = image_shape[2]
        image_shape[2] = temp
        return image_shape
    
    # def make_contour(self, four_channel_image, kernel_size=2, num_colors=4):
    #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #     dilate_image = cv2.dilate(four_channel_image, kernel, iterations=1)
    #     alpha_channel = dilate_image[:, :, 3]
    #     contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     image_with_contour = four_channel_image.copy()

    #     for contour in contours:
    #         color = (0, 0, 0, 255)  
    #         most_sutable_colors = self.find_most_sutable_color(four_channel_image, contours, kenel_size = 2, num_colors = 7)
    #         most_sutable_color = most_sutable_colors[6]
    #         most_sutable_color_reshaped = self.reshape_image(most_sutable_color)
    #         color = np.append(most_sutable_color_reshaped, 255)
    #         cv2.drawContours(image_with_contour, [contour], -1, color, thickness=1)
            
    #     return image_with_contour
    # def select_color(self, four_channel_image, contours):
    #     color = (0, 0, 0, 255)  
    #     most_sutable_colors = self.find_most_sutable_color(four_channel_image, contours, kenel_size = 2, num_colors = 7)
    #     filtered_colors = []
    #     for color in most_sutable_colors:
    #         if not np.array_equal(color, [0, 0, 0]):
    #             filtered_colors.append(color)
    #     darkest_index = np.argmax(filtered_colors)
    #     most_sutable_color = filtered_colors[darkest_index]
    #     most_sutable_color = most_sutable_color * 0.9
    #     most_sutable_color_reshaped = self.reshape_image(most_sutable_color)
    #     print(most_sutable_colors)
    #     color = np.append(most_sutable_color_reshaped, 255)
    #     return color
    
    # def select_color(self, four_channel_image, contours):
    #     color = (0, 0, 0, 255)  
    #     most_sutable_colors = self.find_most_sutable_color(four_channel_image, contours, num_colors=7)
    #     filtered_colors = []

    #     for c in most_sutable_colors:
    #         if not np.array_equal(c, [0, 0, 0]):
    #             filtered_colors.append(c)
    #     print(filtered_colors)
    #     if filtered_colors:
    #         darkest_index = np.argmin(filtered_colors)
    #         most_sutable_color = filtered_colors[darkest_index]
    #         most_sutable_color = most_sutable_color * 0.9
    #         most_sutable_color_reshaped = self.reshape_image(most_sutable_color)
    #         color = np.append(most_sutable_color_reshaped, 255)
    #     print(color)
    #     return color
    
    def select_color(self, four_channel_image, contours):
        color = (0, 0, 0, 255)  
        most_sutable_colors = self.find_most_sutable_color(four_channel_image, contours, num_colors=7)
        filtered_colors = []

        for c in most_sutable_colors:
            if not np.array_equal(c, [0, 0, 0]):
                filtered_colors.append(c)
        print(filtered_colors)
        if filtered_colors:
            darkest_index = np.argmin(filtered_colors)
            if darkest_index < len(filtered_colors):
                most_sutable_color = filtered_colors[darkest_index]
                most_sutable_color = most_sutable_color - most_sutable_color * 0.1
                most_sutable_color = np.clip(most_sutable_color, 0, 255)
                most_sutable_color_reshaped = self.reshape_image(most_sutable_color)
                color = np.append(most_sutable_color_reshaped, 255)

        print(f"Selected color: {color}")
        return color

    def make_contour(self, four_channel_image, kernel_size_to_dilate=1, kernel_size_to_smooth=5):
        
        
        kernel_to_dilate = np.ones((kernel_size_to_dilate, kernel_size_to_dilate), np.uint8)
        dilate_image = cv2.dilate(four_channel_image, kernel_to_dilate, iterations=1)
        
        smoothed_image = cv2.medianBlur(dilate_image, kernel_size_to_smooth) 
        print(smoothed_image.shape)
        bgr = smoothed_image[:, :, :3]
        alpha_channel = smoothed_image[:, :, 3]

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_with_alpha = cv2.merge((gray, gray, gray, alpha_channel))

        image_with_contour = four_channel_image.copy()

        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_with_alpha[:, :, 0])
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        image_with_contour[mask == 0] = [0, 0, 0, 0]
        color = self.select_color(four_channel_image, contours)

        cv2.drawContours(image_with_contour, contours, -1, color, thickness=1)

        contour_without_image = np.zeros_like(image_with_contour)
        cv2.drawContours(contour_without_image, contours, -1, color, thickness=1)

        cv2.imwrite("/home/nikolay/aseprite/image_data/whale_second_process/image_with_contour_star_fish.png", image_with_contour)
        cv2.imwrite("/home/nikolay/aseprite/image_data/whale_second_process/image_without_contour_7_star_fish.png", contour_without_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return image_with_contour, contour_without_image

    def find_mode(self, neighbors):
        if len(neighbors) == 0:
            return None
        neighbors_array = np.array(neighbors)
        mode_result = stats.mode(neighbors_array, axis=0)
        mode_color = mode_result.mode[0]
        count = mode_result.count[0]
        if np.all(count > 1):
            return mode_color.astype(int)
        return None
    
    def replace_outliers_with_mode(self, img, count_of_neighbors = 8, threshold = 200):
        flag = 0
        output_img = img.copy()
        rows, cols = img.shape[:2]
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                current_pixel = img[y, x][:3]
                neighbors = [img[ny, nx][:3] for dy in range(-1, 2) for dx in range(-1, 2)
                            if not (dx == 0 and dy == 0)
                            for ny, nx in [(y + dy, x + dx)]
                            if 0 <= nx < cols and 0 <= ny < rows]
                for neighbor in neighbors:
                    diff = np.mean(np.abs(neighbor - current_pixel), axis=0).sum()
                    if diff > threshold:
                        flag +=1
                    if flag >= count_of_neighbors:
                        mode_color = self.find_mode(neighbors)
                        if mode_color is not None:
                            output_img[y, x][:3] = mode_color 
        return output_img
        
    
    def apply_KNN(self, img, x, y, k = 3):
        neighbors = []
        dx = dy = 1
        for ny in range(y-dy, y+dy+1):
            for nx in range(x-dx, x+dx+1):
                if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] and not (nx == x and ny == y):
                    if img[ny, nx, 3] != 0:
                        neighbors.append(img[ny, nx, :3])
        
        if len(neighbors) > 0:
            return np.mean(neighbors, axis=0)
        else:
            return img[y, x, :3]

    def remove_artifacts(self, img):
        alpha_channel = img[:, :, 3]
        contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(alpha_channel)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] == 255 and img[y, x, 3] == 0:
                    # img[y, x] = [255, 255, 255, 255]
                    temp = self.apply_KNN(img, x, y)
                    img[y, x] = np.append(temp, [255])
        return img
    
    def convert_png_to_svg(self, image, svg_path):
        width, height, _ = image.shape
        svg_data = f'<svg width="{width}" height="{height}" version="1.1" xmlns="http://www.w3.org/2000/svg">\n'
        
        for y in range(height):
            for x in range(width):
                b, g, r, a = image[y,x]
                if a != 0:
                    rgba_color = f'rgba({r}, {g}, {b}, {a})'
                    svg_data += f'<rect x="{x}" y="{y}" width="1" height="1" fill="{rgba_color}" />\n'
                    
        svg_data += '</svg>'
        
        with open(svg_path, 'w') as f:
            f.write(svg_data) 