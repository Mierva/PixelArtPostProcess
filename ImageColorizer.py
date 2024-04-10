from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from keras.models import Sequential
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class ImageColorizer:
    def __init__(self, image_shape):
        self.model = self._build_model()        
        self.image_shape = image_shape        

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(None, None, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    # @staticmethod
    def process_image(self, img_path=0, target_size=None):
        if target_size is None:
            target_size = self.image_shape
            
        img = load_img(img_path) 
        img = img.resize(target_size, Image.BILINEAR)
        image = img_to_array(img) 
        image = np.array(image, dtype=float)
        lab = rgb2lab(1.0/255*image)
        X = lab[:, :, 0]
        Y = lab[:, :, 1:] / 128 
     
        X = X.reshape(1, target_size[0], target_size[1], 1)
        Y = Y.reshape(1, target_size[0], target_size[1], 2)
        return X, Y

    def train_model(self, img_path, epochs=200):
        X, Y = self.process_image(img_path)
        self.model.fit(x=X, y=Y, batch_size=1, epochs=epochs)

    def colorize_image(self, img_path):
        X, _ = self.process_image(img_path)
        output = self.model.predict(X)*128          
        ab = np.clip(output[0], -128, 127)

        size = X.shape[1:3] 

        cur = np.zeros((size[0], size[1], 3))
        cur[:,:,0] = np.clip(X[0][:,:,0], 0, 100)
        cur[:,:,1:] = ab
        
        return Image.fromarray((lab2rgb(cur) * 255).astype(np.uint8))