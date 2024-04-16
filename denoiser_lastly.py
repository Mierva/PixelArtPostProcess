import cv2
import numpy as np
from scipy import stats

# def find_mode(neighbors):
#     if len(neighbors) == 0:
#         return None
#     neighbors_array = np.array(neighbors)
#     mode_result = stats.mode(neighbors_array, axis=0)
#     mode_color = mode_result.mode[0]
#     count = mode_result.count[0]
#     if np.all(count > 1):
#         return mode_color.astype(int)
#     return None

def most_frequent_color(neighbors):
    neighbors_array = np.vstack(neighbors)  # Преобразование списка в массив
    unique, counts = np.unique(neighbors_array, axis=0, return_counts=True)
    most_frequent_index = np.argmax(counts)
    return unique[most_frequent_index]

# def replace_outliers_with_mode(img, count_of_neighbors, threshold):
#     flag = 0
#     output_img = img.copy()
#     rows, cols = img.shape[:2]
#     for y in range(1, rows - 1):
#         for x in range(1, cols - 1):
#             current_pixel = img[y, x][:3]
#             neighbors = [img[ny, nx][:3] for dy in range(-1, 2) for dx in range(-1, 2)
#                          if not (dx == 0 and dy == 0)
#                          for ny, nx in [(y + dy, x + dx)]
#                          if 0 <= nx < cols and 0 <= ny < rows]
#             for neighbor in neighbors:
#                 diff = np.abs(neighbor - current_pixel)
#                 # if diff > threshold:
#                 #     print(diff)
#                 # if diff > threshold:
#                 #     flag +=1
#                 if np.sum(diff > threshold) > count_of_neighbors:
#                     if flag >= count_of_neighbors:
#                         print(flag)
#                         mode_color = most_frequent_color(neighbors)
#                         if mode_color is not None:
#                             output_img[y, x][:3] = mode_color 
#     return output_img

# def replace_outliers_with_mode(img, count_of_neighbors, threshold):
#     output_img = img.copy()
#     rows, cols = img.shape[:2]
#     for y in range(1, rows - 1):
#         for x in range(1, cols - 1):
#             current_pixel = img[y, x][:3]
#             neighbors = [img[ny, nx][:3] for dy in range(-1, 2) for dx in range(-1, 2)
#                          if not (dx == 0 and dy == 0)
#                          for ny, nx in [(y + dy, x + dx)]
#                          if 0 <= nx < cols and 0 <= ny < rows]
#             flag = 0  
#             for neighbor in neighbors:
#                 diff = np.abs(neighbor - current_pixel)
#                 if np.sum(diff > threshold) > count_of_neighbors:
#                     flag += 1
#                     if flag >= count_of_neighbors:
#                         print(f"Flag: {flag}")
#                         mode_color = most_frequent_color(neighbors)
#                         if mode_color is not None:
#                             output_img[y, x][:3] = mode_color 
#     return output_img

def replace_outliers_with_mode(img, count_of_neighbors, threshold):
    output_img = img.copy()
    rows, cols = img.shape[:2]
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            current_pixel = img[y, x][:3]
            neighbors = [img[ny, nx][:3] for dy in range(-1, 2) for dx in range(-1, 2)
                         if not (dx == 0 and dy == 0)
                         for ny, nx in [(y + dy, x + dx)]
                         if 0 <= nx < cols and 0 <= ny < rows]
            flag = 0  
            for neighbor in neighbors:
                diff = np.mean(np.abs(neighbor - current_pixel), axis=0).sum()
                if diff > threshold:
                    flag += 1
                    if flag >= count_of_neighbors:
                        print(flag)
                        mode_color = most_frequent_color(neighbors)
                        if mode_color is not None:
                            output_img[y, x][:3] = mode_color 
                        break  
    return output_img

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    iterations = 5
    count_of_neighbors = [4, 5, 5, 4, 4]
    threshold = [100, 120, 100, 120, 100]
    image = img.copy()
    for i in range(iterations):
        result_img = replace_outliers_with_mode(image, count_of_neighbors[i], threshold[i])
        cv2.imwrite(f'/home/nikolay/aseprite/working_code/production/images/result{i}.png', result_img)
        image = result_img

process_image('/home/nikolay/aseprite/working_code/production/images/bober_pp_epoch200_psize6_ccolor2_neighbors[4]_thresh[150]_iters1.png')

