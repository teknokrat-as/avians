
import numpy as np
import gc
import os

class AutoencoderDataset(): 
    
    def __init__(self, load_from=None, 
                 initial_size=1000,
                 img_ch=1, 
                 img_x=64,
                 img_y=64):

        if load_from: 
            ds = np.load(load_from, mmap_mode='r')
            self.component_images = ds['component_images']
            self.new_component_index = ds['new_component_index']
            ds = None
            gc.collect()
        else: 
            self.component_images = np.zeros(shape=(initial_size, img_ch, img_x, img_y),
                                             dtype=np.uint8)
            self.new_component_index = 0

    def add_component(self, component_img): 
        self.component_images[self.new_component_index] = component_img
        self.new_component_index += 1
        current_max_size = self.component_images.shape[0]
        if self.new_component_index >= current_max_size:
            self.component_images = np.append(self.component_images, np.zeros_like(self.component_images), axis=0)


    def add_components(self, component_arr): 
        n, ch, x, y = component_arr.shape
        current, last = (self.new_component_index, self.new_component_index + n)
        if last >= self.component_images.shape[0]: 
            self.component_images = np.append(self.component_images, np.zeros_like(self.component_images), axis=0)

        self.component_images[current:last] = component_arr
        self.new_component_index = last

    def as_dataset(self): 
        x_train = self.component_images[:self.new_component_index]
        return x_train.astype('float32') / 255

    def write(self, filename):
        np.savez_compressed(filename, 
                            component_images=self.component_images,
                            new_component_index=self.new_component_index)
