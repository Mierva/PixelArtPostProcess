# Postprocessing for Pixel Art:

## Utilizing Aseprite for crafting pixel art images, then enhancing their quality through algorithmic refinement


### This README outlines the usage, installation, and functionality of a postprocessing tool designed for enhancing pixel art images. This tool encompasses a range of operations such as:
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

### Installation
To set up the postprocessing tool, you will need to ensure that your environment is ready and all dependencies are installed.

### Prerequisites

An environment capable of running Jupyter Notebooks if modifications or interactive experimentation are desired.
### Dependencies
The tool relies on several external libraries which need to be installed via pip:

pip install opencv-python numpy cairosvg Pillow keras scikit-image Pylette matplotlib
Ensure all dependencies are installed to avoid any runtime errors.

### Usage
The tool can be executed from the command line with various options to customize the processing according to your needs.

### Basic Command Structure

python postprocess_pixel_art.py --input_image_path INPUT_PATH --output_image_svg OUTPUT_PATH [OPTIONS]

### Required Arguments
--input_image_path: The path to the input image file (SVG or PNG format).
--output_image_svg: The path where the output SVG image will be saved.

### Optional Arguments
--iterations: Number of iterations for postprocessing algorithms. Default is 1.
--threshold: Threshold for delta between pixel values. Default is 150.
--neighbors: Number of neighbors to consider for delta calculation. Default is 4.
--contour_color: Index for the contour color in the final image. Default is 1.
--image_shape: Shape of the output image (width, height, channels). Default is [128, 128, 3].
--palette_size: Number of colors in the final image palette. Default is 10.
--epochs: Number of epochs for training the colorization model. Default is 2.
--mode: Mode for color extraction (KM for K-Means, MC for Median Cut). Default is 'MC'.

### Example Command

python postprocess_pixel_art.py --input_image_path './images/pixel_art.svg' --output_image_svg './output/colored_art.svg' --iterations 2 --threshold 100 --neighbors 3 --palette_size 8 --epochs 10

### Features and Functions
The tool consists of two main classes, ImageProcessor and ImageColorizer, alongside utility functions for color processing.

ImageProcessor
Handles image format conversions, isolating pixel changes, contour enhancements, and artifact removal. It offers functions like:

- from_svg_to_png(): Converts SVG images to PNG.
- change_isolating_pixels(): Modifies isolated pixels based on neighbors.
- make_contour(): Enhances contours in the image.
- remove_artifacts(): Removes unwanted artifacts from the image.
- ImageColorizer
- Utilizes a convolutional neural network to colorize grayscale images automatically. Key methods include:

_build_model(): Constructs the CNN model for colorization.
train_model(): Trains the model on the given image.
colorize_image(): Applies colorization to the image.
### Utility Functions
process_image_colors(): Applies a custom color palette to the image.
hex_to_rgb(): Converts HEX color codes to RGB format.
## Conclusion
This tool offers a streamlined approach to enhance and colorize pixel art images, making it suitable for artists and developers looking to refine their pixel art with minimal manual intervention. By automating the postprocessing steps, it enables users to quickly produce high-quality, vibrant pixel art images ready for use in various applications.
