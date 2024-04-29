from PIL import Image

def resize_function(input_path, output_path, scale_factor):
    image = Image.open(input_path)
    width, height = image.size
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    if new_width % 2 != 0:
        new_width += 1
    if new_height % 2 != 0:
        new_height += 1
        
    resize_image = image.resize((new_width, new_height), Image.LANCZOS)
    resize_image.save(output_path)
    resized_width, resized_height = resize_image.size
    return resized_width, resized_height
    

if __name__ == '__main__':
    resize_function('/home/nikolay/aseprite/image_data/car/original_car.png', '/home/nikolay/aseprite/image_data/car/resized_car.png', 0.21)
