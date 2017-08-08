"""This file defines a base class for Avians component models. It provides a
common framework to instantiate models and preprocess images. Each child class
can implement the functions provided here."""

from abc import ABCMeta, abstractmethod

class AviansModel(metaclass=ABCMeta):


    def __init__(self, weight_file):
        self.weight_file = weight_file
        self.create_model()
        self.load_weights()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def prepare_binary_image(self, image): 
        pass
        
    def prepare_image(self, image): 
        if len(image.shape) == 2: 
            ## TODO: WRITE a detect_edge function to call for grayscale image
            return prepare_binary_image(detect_edge(image))
        elif len(image.shape) == 3:
            ## TODO: WRITE a color_to_grayscale function
            return prepare_binary_image(detect_edge(color_to_grayscale(image)))
        else: 
            raise Exception("Image should have 2 or 3 dimensions")

    def classify_image_array(self, image_array):
        """Classifies the given binary image array
        
        @param image_array: A n*w*h image array that contains 
        """
        image_array_classes = []
        
        n, w, h = image_array.shape
        for i in range(n): 
            image_array_classes.append(self.classify_image(image_array[i]))
        return image_array_classes    
        
            
    @abstractmethod
    def classify_image(self, image): 
        pass
        
    
    def __call__(self, image): 
        return self.classify_image(image)
