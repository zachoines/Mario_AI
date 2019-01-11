# Image Preprocessing

# Importing the libraries
import numpy as np
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box

# Preprocessing the Images

class GrayScaleImage(ObservationWrapper):
    
    def __init__(self, env, height = 64, width = 64, grayscale = True, crop = lambda img: img):
        super(GrayScaleImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_channels = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [height, width, n_channels])

    def _observation(self, img):
        img = self.crop(img)
        img = imresize(img, self.img_size)

        # Convert to grayscale if enabled using mean method
        if self.grayscale:
            img = np.mean(img, axis=-1,keepdims=1)
            # img = np.max(img, axis = -1, keepdims = 1) / 2 +  np.min(img, axis = -1, keepdims = 1) / 2

        return img




        # img = self.crop(img)
        # img = tf.image.resize_images(img, self.img_size)
        # if self.grayscale:
        #     img = tf.image.rgb_to_grayscale(img)
        # return img

