from Pylette import extract_colors
from ImageColorizer import ImageColorizer
from ImageProcessor import ImageProcessor
from pathlib import Path
import numpy as np
import argparse
import cv2
import os
    
    
def process_image_colors(input_img, output_image_path, palette_size=10, 
                         indexes_for_drop=[], resize=True, mode="MC",sort_mode='luminance'):
    input_image = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)

    if input_image.shape[2] == 4:  
        bgr = input_image[:, :, :3]  
        alpha_channel = input_image[:, :, 3] 
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
         

    palette = extract_colors(image=input_img, palette_size=palette_size, 
                             resize=False, 
                             #mode=mode, 
                             sort_mode=sort_mode)

    palette_colors = np.array([color.rgb for color in palette])
    
    indexes_for_drop.extend(np.where((palette_colors == [0, 0, 0]).all(axis=1))[0])
    # indexes_for_drop.extend(np.where((palette_colors == [21, 13, 9]).all(axis=1))[0])
    palette_colors = np.delete(palette_colors, indexes_for_drop, axis=0)
    print(palette_colors)
    
    palette_colors_reshaped = palette_colors.reshape((-1, 1, 1, 3))
    distances = np.sqrt(((rgb.reshape((1,) + rgb.shape) - palette_colors_reshaped) ** 2).sum(axis=-1))
    nearest_colors_index = np.argmin(distances, axis=0)
    processed_image = palette_colors[nearest_colors_index]
            
    processed_bgr = cv2.cvtColor(processed_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contour = processed_bgr.copy()

    color = tuple(map(int, palette_colors[np.argmin(np.sum(np.array(palette_colors), axis=1))]))
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    print(bgr_color)
    # darker_bgr_color = tuple(int(c * 1) for c in bgr_color)
    darker_bgr_color = (179, 208, 234)
    print(darker_bgr_color)
    cv2.drawContours(image_with_contour, contours, -1, darker_bgr_color, thickness=1)
    if input_image.shape[2] == 4:
        processed_image_with_alpha = cv2.merge((image_with_contour, alpha_channel))
                
        cv2.imwrite(output_image_path, processed_image_with_alpha)
        return processed_image_with_alpha
    else:
        cv2.imwrite(output_image_path, image_with_contour)
        return image_with_contour
    
def clear_tmp_imgs(folder, args, output_path_png):    
    for i in range(args.iterations):
        os.remove(os.path.join(folder, f'./output_circle{i+1}.png'))
        os.remove(os.path.join(folder, f'./colored_circle{i+1}.png'))
        
    if args.output_img.endswith('.svg'):
        os.remove(args.output_img.replace('.svg','.png'))
    else:
        os.remove(args.output_img)
    
    filename = Path(args.input_img).stem
    os.remove(args.input_img.replace('.svg', '.png'))    
    os.remove(output_path_png)  

import ast

def convert_threshold(data):
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except (ValueError, SyntaxError) as e:
            print(f"Ошибка при конвертации: {e}")
            return []

    if isinstance(data, list):
        try:
            return [int(num) for num in data]
        except ValueError as e:
            print(f"Ошибка при конвертации чисел в списке: {e}")
            return []
    else:
        raise ValueError("Входные данные должны быть списком чисел или строкой, представляющей список.")


def main(args):    
    folder = './images'    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    input_img = args.input_img
    if args.input_img.endswith('.svg'):        
        processor = ImageProcessor(input_img)
        processor.from_svg_to_png()
    else:
        # processor = ImageProcessor(args.image_path)
        processor = ImageProcessor(input_img)
    
    print(f'\n{args.image_shape}|{type(args.image_shape)}\n')
    image_colorizer = ImageColorizer(args.image_shape)
    image_colorizer.train_model(processor.output_png_image_path, epochs=args.epochs)
    
    
    thresholds = convert_threshold(args.threshold)
    neighbors = convert_threshold(args.neighbors)
    for i in range(args.iterations):
        print(thresholds[i])
        print(neighbors[i])
        processor.update_processing_parameters(thresholds[i], neighbors[i])
        processor.change_isolating_pixels()
        
        output_path = os.path.join(folder, f'./output_circle{i+1}.png')        
        gray_image, _ = processor.save_modified_image(output_path)

        result_image = image_colorizer.colorize_image(output_path)
        circle_path = os.path.join(folder, f'./colored_circle{i+1}.png')
        result_image.save(circle_path)
        result_image_npp = np.array(result_image)
        result_image_np_bgr = cv2.cvtColor(result_image_npp, cv2.COLOR_RGB2BGR)

    alpha_channel = processor.four_channel_png_to_bgr()
    four_channel_image = processor.convert_from_3_to_4_channel_image(result_image_np_bgr, alpha_channel)
    # print(four_channel_image.shape)
    # cv2.imwrite("/home/nikolay/aseprite/image_data/whale_second_process/image_without_contour_1111.png", four_channel_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # TODO: add optinal denoiser
    # image_denoise = processor.replace_outliers_with_mode(four_channel_image, count_of_neighbors = 4, threshold = 100)
    # cv2.imwrite("/home/nikolay/aseprite/image_data/whale_second_process/image_without_contour_2222.png", four_channel_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    image_with_contour, _  = processor.make_contour(four_channel_image, kernel_size_to_dilate=1, kernel_size_to_smooth=5)
    image_without_artifacts = processor.remove_artifacts(image_with_contour)
    # cv2.imwrite("/home/nikolay/aseprite/image_data/whale_second_process/image_without_contour_3333.png", image_denoise, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    output_path_png = os.path.join(folder, './output.png')
    
    cv2.imwrite(output_path_png, image_without_artifacts, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    output_image_png = args.output_img.replace('.svg', '.png')    
    
    filename = Path(output_image_png.replace(Path(output_image_png).suffix, ''))
    final_image_path = f"{filename}_epoch{args.epochs}_psize{args.palette_size}_ccolor{args.contour_color}_neighbors{args.neighbors}_thresh{args.threshold}_iters{args.iterations}.png"
    
    output_image = process_image_colors(output_path_png, final_image_path, 
                                        palette_size=args.palette_size, indexes_for_drop=[0, 2],
                                        resize=True, mode=args.mode, sort_mode='luminance')
    filename = Path(args.output_img.replace(Path(args.output_img).suffix, ''))
    final_image_path = f"{filename}_epoch{args.epochs}_psize{args.palette_size}_ccolor{args.contour_color}_neighbors{args.neighbors}_thresh{args.threshold}_iters{args.iterations}.svg"
    processor.convert_png_to_svg(output_image, final_image_path)
    
    # clear_tmp_imgs(folder, args, output_path_png)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocessing for pixel art images')
    
    parser.add_argument('--input_img', type=str, required=True, help='Path to image svg/png format')
    parser.add_argument('--output_img', type=str, required=True, help='Path to output svg colored image')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for postprocessing algorithm')
    parser.add_argument('--threshold', nargs='+', type=int, default=150, help='Threshold for delta between pixel values')
    parser.add_argument('--neighbors', nargs='+', type=int, default=4, help='Number of neighbors for delta between pixel values')    
    parser.add_argument('--contour_color', type=int, default=1, help='Color for contour in range[0-6]')
    parser.add_argument('--image_shape', type=int, nargs=2, default=(128, 128), help='Shape of output image such as: (128, 128, 3), (256, 256, 3), (512, 512, 3)')
    parser.add_argument('--palette_size', type=int, default=10, help='Number of colors in palette')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training model')
    parser.add_argument('--mode', type=str, default='MC', help='Mode for processing: (KM/MC)')

    args = parser.parse_args()
    
    main(args)
    
