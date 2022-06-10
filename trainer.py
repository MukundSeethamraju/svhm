import tensorflow as tf
import datetime

from data_loader import ProcessMatFile, TfRecordDataset
from preprocessing import Preprocessing
from model import SimpleModel, SimpleModel2
from metrics import CustomLoss, ClassificationAccuracy

BATCH_SIZE = 256
TRAIN_DATA_PATH = '/Users/mukund/Documents/svhm/data/train/digitStruct.mat'
TEST_DATA_PATH = '/Users/mukund/Documents/svhm/data/test/digitStruct.mat'
IMG_SIZE = (128,128)


class Trainer():
    def __init__(self, train_data_path, test_data_path, batch_size, img_size):
        self._batch_size = batch_size
        self._img_size = img_size
        self.preprocessor = Preprocessing(resize=img_size)
        self.dataset_builder = TfRecordDataset(train_data_path, test_data_path, batch_size=batch_size)
    
    def build_model(self, kernel_size, nb_filter, pool_size, activation, format='classification'):
        num_classes = self.dataset_builder.num_classes
        if format == 'classification':
            num_neurons = num_classes
        elif format == 'detection':
            num_neurons = num_classes + 4
        self.model = SimpleModel2(kernel_size, nb_filter, pool_size, activation, num_neurons)
        
    def train(self, format='classification', optimizer='adam', metrics=['accuracy'], epochs=20, callback=False):
        train_dataset, validation_dataset = self.get_datasets(format)
        loss_fn = self.get_loss(format)
        if format == 'detection':
            metrics = [ClassificationAccuracy]
        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)
        if callback:
            callback = self.add_callback()
            self.model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[callback])
        else:
            self.model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
            
    def get_datasets(self, format):
        train_dataset = self.dataset_builder.get_dataset(self.preprocessor, 'train', format)
        validation_dataset = self.dataset_builder.get_dataset(self.preprocessor, 'validation', format)
        return train_dataset, validation_dataset

    def save_model(self, location):
        self.model.save(location, overwrite=False)
        
    def create_datasets(self):
        self.dataset_builder.create_datasets()
        
    def add_callback(self):
        log_dir = "/Users/mukund/Documents/svhm/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return tensorboard_callback
    
    def get_loss(self, format):
        if format == 'classification':
            return 'CategoricalCrossentropy'
        elif format =='detection':
            num_classes = self.dataset_builder.num_classes
            return CustomLoss(num_classes)