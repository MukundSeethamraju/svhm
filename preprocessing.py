import tensorflow as tf
from tf_image.application.augmentation_config import AugmentationConfig,  AspectRatioAugmentation
from tf_image.application.tools import random_augmentations
import numpy as np

class Preprocessing():
    
    def __init__(self, training=True, normalize=True, resize=[128,128], augment=False, convert_to_greyscale=True):
        self._training = training
        self._normalize = normalize
        self._resize = resize
        self._augment = augment
        self._convert_to_greyscale = convert_to_greyscale
        
    def augment(self, features):
        config = AugmentationConfig()
        config.crop = False
        config.distort_aspect_ratio = AspectRatioAugmentation.NONE
        config.erasing = False
        config.quality = False
        config.rotate_max = 0
        image = features.get('image')
        if self._format == 'detection':
            image_shape = image.get_shape()
            bounding_boxes = features.get('bounding_boxes')
            if not self._resize:
                bounding_boxes = tf.reshape(bounding_boxes, (-1,4))
            bounding_boxes /= tf.cast(
                tf.stack([image_shape[0], image_shape[1], image_shape[0], image_shape[1]]), tf.float32)
            image_augmented, bboxes_augmented = random_augmentations(image, config, bboxes=bounding_boxes)
            features['bounding_boxes'] = bboxes_augmented
        else:
            image_augmented = random_augmentations(image, config)
        features['image'] = image_augmented
        return features
        
    def normalize(self, features):
        image = tf.cast(features.get('image'), tf.float32)
        image /= 255.0
        image = tf.image.per_image_standardization(image)
        features['image'] = image
        if self._format == 'detection':
            image_shape = features.get('image_shape')
            bounding_boxes = features.get('bounding_boxes')
            bounding_boxes = tf.reshape(bounding_boxes, (-1,4))
            bounding_boxes /= tf.cast(
                tf.stack([image_shape[0], image_shape[1], image_shape[0], image_shape[1]]), tf.float32)
            features['bounding_boxes'] = bounding_boxes
        return features
    
    def resize(self, features):
        features['image'] = tf.image.resize(features['image'], size=self._resize)
        if self._format == 'detection':
            image_shape = features.get('image_shape') 
            bounding_boxes = features.get('bounding_boxes')
            if not self._normalize:
                bounding_boxes = tf.reshape(bounding_boxes, (-1,4))
                bounding_boxes /= tf.cast(
                    tf.stack([self._resize[0]/image_shape[0], self._resize[1]/image_shape[1], self._resize[0]/image_shape[0], self._resize[1]/image_shape[1]]),
                    tf.float32)
            features['bounding_boxes'] = bounding_boxes
        return features
    
    def greyscale(self, features):
        features['image'] = tf.image.rgb_to_grayscale(features['image'])
        return features
        
    def get_preprocessing_fn(self, mode, format):
        self._format = format
        if mode != 'train':
            self._training = False
        fn = self.id
        if self._convert_to_greyscale:
            fn = self.compose(self.greyscale, fn)
        if self._normalize:
            fn = self.compose(self.normalize, fn)
        if self._resize:
            fn = self.compose(self.resize, fn)
        if self._augment and self._training:
            fn = self.compose(self.augment, fn)
        return fn
        
    def id(self, x):
        return x
        
    def compose(self, *functions):
        def inner(arg):
            for f in reversed(functions):
                arg = f(arg)
            return arg
        return inner