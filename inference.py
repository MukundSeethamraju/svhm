import tensorflow as tf
import datetime
import numpy as np

from data_loader import ProcessMatFile, TfRecordDataset
from preprocessing import Preprocessing
from model import SimpleModel, SimpleModel2
from metrics import CustomLoss, ClassificationAccuracy

BATCH_SIZE = 256
TRAIN_DATA_PATH = '/Users/mukund/Documents/svhm/data/train/digitStruct.mat'
TEST_DATA_PATH = '/Users/mukund/Documents/svhm/data/test/digitStruct.mat'
IMG_SIZE = (128,128)


class Inference():
    def __init__(self, train_data_path, test_data_path, batch_size, img_size):
        self._batch_size = batch_size
        self._img_size = img_size
        self.preprocessor = Preprocessing(resize=img_size)
        self.dataset_builder = TfRecordDataset(train_data_path, test_data_path, batch_size=batch_size)
    

    def get_dataset(self, format='classification'):
        validation_dataset = self.dataset_builder.get_dataset(self.preprocessor, 'validation', format)
        return validation_dataset

    def load_model(self, location):
        self.model = tf.keras.models.load_model(location)
        return self.model
        
    
    def infer_on_sample(self, image, label):
        output = self.model(image, training=False)
        predictions = [np.argmax(i.numpy()) for i in output]
        true_labels = [np.argmax(i.numpy()) for i in label]
        return predictions, true_labels, image