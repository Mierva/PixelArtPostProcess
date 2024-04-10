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
                             resize=resize, mode=mode, sort_mode=sort_mode)

    palette_colors = np.array([color.rgb for color in palette])
    
    palette_colors = np.delete(palette_colors, indexes_for_drop, axis=0)
    
    palette_colors_reshaped = palette_colors.reshape((-1, 1, 1, 3))
    distances = np.sqrt(((rgb.reshape((1,) + rgb.shape) - palette_colors_reshaped) ** 2).sum(axis=-1))
    nearest_colors_index = np.argmin(distances, axis=0)
    processed_image = palette_colors[nearest_colors_index]
            
    processed_bgr = cv2.cvtColor(processed_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if input_image.shape[2] == 4:
        processed_image_with_alpha = cv2.merge((processed_bgr, alpha_channel))        
        cv2.imwrite(output_image_path, processed_image_with_alpha)
        return processed_image_with_alpha
    else:
        cv2.imwrite(output_image_path, processed_bgr)
        return processed_bgr
    
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
    
    for i in range(args.iterations):
        processor.update_processing_parameters(args.threshold, args.neighbors)
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
    
    image_with_contour = processor.make_contour(four_channel_image, kernel_size=2)
    image_without_artifacts = processor.remove_artifacts(image_with_contour)
    output_path_png = os.path.join(folder, './output.png')
    
    cv2.imwrite(output_path_png, image_without_artifacts, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    output_image_png = args.output_img.replace('.svg', '.png')    
    
    output_image = process_image_colors(output_path_png, output_image_png, 
                                        palette_size=args.palette_size, indexes_for_drop=[0, 2],
                                        resize=True, mode=args.mode, sort_mode='luminance')
    
    processor.convert_png_to_svg(output_image, args.output_img)
    
    clear_tmp_imgs(folder, args, output_path_png)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocessing for pixel art images')
    
    parser.add_argument('--input_img', type=str, required=True, help='Path to image svg/png format')
    parser.add_argument('--output_img', type=str, required=True, help='Path to output svg colored image')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations for postprocessing algorithm')
    parser.add_argument('--threshold', type=int, default=150, help='Threshold for delta between pixel values')
    parser.add_argument('--neighbors', type=int, default=4, help='Number of neighbors for delta between pixel values')
    parser.add_argument('--contour_color', type=int, default=1, help='Color for contour in range[0-6]')
    parser.add_argument('--image_shape', type=int, nargs=2, default=(128, 128), help='Shape of output image such as: (128, 128, 3), (256, 256, 3), (512, 512, 3)')
    parser.add_argument('--palette_size', type=int, default=10, help='Number of colors in palette')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training model')
    parser.add_argument('--mode', type=str, default='MC', help='Mode for processing: (KM/MC)')

    args = parser.parse_args()
    
    main(args)
    
