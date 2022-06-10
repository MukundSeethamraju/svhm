import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        loss_classification_fn = tf.nn.softmax_cross_entropy_with_logits
        # loss_detection_fn = tf.keras.losses.MeanSquaredError()
        loss_detection_fn = tfa.losses.GIoULoss()
        classification_loss = loss_classification_fn(y_true[:,:self.num_classes], y_pred[:,:self.num_classes])
        localization_loss = loss_detection_fn(y_true[:,self.num_classes:], y_pred[:,self.num_classes:])
        loss = classification_loss + 5*localization_loss
        return loss
        
        
def ClassificationAccuracy(y_true, y_pred):
        return tf.keras.metrics.categorical_accuracy(y_true[:,:-4], tf.nn.softmax(y_pred[:,:-4]))

    