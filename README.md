# Postprocessing for Pixel Art:

Utilizing Aseprite for crafting pixel art images, then enhancing their quality through algorithmic refinement


### Installation
`conda env create -f environment.yaml`

### Basic Command Structure
`python postprocess_pixel_art.py --input_img INPUT_PATH --output_img OUTPUT_PATH [OPTIONS]`

### Example Command
`python main.py --input_img fish.svg --output_img ./images/bober_pp.svg --iterations 1 --threshold 150 --neighbors 4 --contour_color 1 --palette_size 10 --epochs 10 --mode MC`

### Required Arguments
`--input_img`: The path to the input image file (SVG format).
`--output_img`: The path where the output SVG image will be saved.

### Optional Arguments
`--iterations`: Number of iterations for postprocessing algorithms. Default is 1.<br>
`--threshold`: Threshold for delta between pixel values. Default is 150.<br>
`--neighbors`: Number of neighbors to consider for delta calculation. Default is 4.<br>
`--contour_color`: Index for the contour color in the final image. Default is 1.<br>
`--image_shape`: Shape of the output image (width, height, channels). Default is (128, 128).<br>
`--palette_size`: Number of colors in the final image palette. Default is 10.<br>
`--epochs`: Number of epochs for training the colorization model. Default is 2.<br>
`--mode`: Mode for color extraction (KM for K-Means, MC for Median Cut). Default is 'MC'.<br>




### Postprocessing steps:
- format conversion
- colorization
- artifact removal
- contour improvement to refine pixel art automatically.

### Overview
The postprocessing tool for pixel art includes a comprehensive set of features to enhance and colorize pixel art images. Key functionalities include:

Conversion of SVG to PNG format.
Automatic colorization of grayscale images.
Enhancement of image contours and choose different shades.
Reduction of isolated pixel artifacts.
Application of custom color palettes.
Export of processed images to both PNG and SVG formats.


- from_svg_to_png(): Converts SVG images to PNG.
- change_isolating_pixels(): Modifies isolated pixels based on neighbors.
- make_contour(): Enhances contours in the image.
- remove_artifacts(): Removes unwanted artifacts from the image.
- ImageColorizer
- Utilizes a convolutional neural network to colorize grayscale images automatically. Key methods include:
- _build_model(): Constructs the CNN model for colorization.
- train_model(): Trains the model on the given image.
- colorize_image(): Applies colorization to the image.
  
### Utility Functions
- process_image_colors(): Applies a custom color palette to the image.
- hex_to_rgb(): Converts HEX color codes to RGB format.

**Before:**<br>
![image](https://github.com/Kimiko12/Postprocess_for_PixelArt_images/assets/79062452/fd758fbc-44d1-4f1f-a597-0efa20fa5239)<br>
**After:**<br>
![image](https://github.com/Kimiko12/Postprocess_for_PixelArt_images/assets/79062452/7f81c576-9537-479f-844e-100d8fcb8bb6) 
