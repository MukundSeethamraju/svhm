import numpy as np
import os
import tensorflow as tf
import h5py
from collections import defaultdict
import pickle

AUTOTUNE = tf.data.AUTOTUNE


class ProcessMatFile:
    def __init__(self, data_path):
        self.data = h5py.File(data_path, 'r')
        self.image_file_names = self.data['digitStruct']['name']
        self.bounding_boxes = self.data['digitStruct']['bbox']
        self._image_file_dir = os.path.split(data_path)[0]
        
    def _get_path(self, image_id):
        file_name = ''.join([chr(c[0]) for c in self.data[self.image_file_names[image_id][0]][:]])
        path = os.path.join(self._image_file_dir, file_name)
        return path

    def _process_bounding_box(self, object):
        if (len(object) > 1):
            object = [self.data[object[j][0]][0][0] for j in range(len(object))]
        else:
            object = [object[0][0]]
        return object

    def _get_bounding_box(self,n):
        bbox = {}
        bb = self.bounding_boxes[n].item()
        bbox['height'] = self._process_bounding_box(self.data[bb]["height"])
        bbox['label'] = self._process_bounding_box(self.data[bb]["label"])
        bbox['left'] = self._process_bounding_box(self.data[bb]["left"])
        bbox['top'] = self._process_bounding_box(self.data[bb]["top"])
        bbox['width'] = self._process_bounding_box(self.data[bb]["width"])
        return bbox
        
    def _get_image_data(self, image_id, num_of_boxes):
        bounding_box_data = self._get_bounding_box(image_id)
        bboxes = [0]*4*num_of_boxes
        labels = [0]*num_of_boxes
        for i in range(len(bounding_box_data['height'])):
            if i > num_of_boxes-1:
                break
            height = bounding_box_data['height'][i]
            width  = bounding_box_data['width'][i]
            left   = bounding_box_data['left'][i]
            top    = bounding_box_data['top'][i]
            
            xmin = float(left)
            xmax = float(left + width)
            ymax = float(top + height)
            ymin = float(height)
            bbox = [xmin, ymin, xmax, ymax]
            
            label  = bounding_box_data['label'][i]
            try:
                labels[i] = label
            except:
                print(bounding_box_data['label'])
                raise
            bboxes[4*i: 4*i+4] = bbox
        
        image_data = {'path':self._get_path(image_id)}
        image_data['labels'] = labels
        image_data['bounding_boxes'] = bboxes
        return image_data
    
    def get_all_data(self, num_of_boxes):
        for image_id in range(len(self.image_file_names)):
            yield self._get_image_data(image_id, num_of_boxes)

    @property
    def num_samples(self):
        return len(self.image_file_names)



class TfRecordDataset():
    
    def __init__(self,  train_file_path, test_file_path, shuffle=False, batch_size=64, num_of_boxes=5):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_of_boxes = num_of_boxes
        self._metadata = {}
        self._labels = set()
        self._collect_metadata = False
        self._train_file_path = train_file_path
        self._test_file_path = test_file_path
        self.load_metadata()
        
    def create_datasets(self):            
        self._create_dataset(mode = 'train')
        self._create_dataset(mode = 'validation')
        self.save_metadata()
        #TODO: implement test dataset as well 
        
    def _create_dataset(self, mode, num_tfrecords=1):
        tfrecords_dir, path = self._get_tfrecords_dir(mode)
        data_processor = ProcessMatFile(path)
        num_samples = data_processor.num_samples

        if mode == 'validation':
            self.update_metadata({"num_validation_samples": num_samples})
        if mode == 'train':
            self.update_metadata({"num_train_samples": num_samples})

        samples_per_tfrecord = int(num_samples/num_tfrecords)
        sample_generator = data_processor.get_all_data(self.num_of_boxes)
        classes = set()
        for tfrec_num in range(num_tfrecords):
            with tf.io.TFRecordWriter(
                tfrecords_dir + "/images_file_%.2i-%i.tfrec" % (tfrec_num, num_samples)
            ) as writer:
                new_classes = self._write_to_tfrecord(writer, sample_generator, samples_per_tfrecord)
        if self._collect_metadata:
            classes |= new_classes
            self.update_metadata({"classes": classes})
        
    def _get_tfrecords_dir(self, mode):
        if mode == 'train':
            path = self._train_file_path
            self._collect_metadata = True
        elif mode == 'validation':
            path = self._test_file_path
            self._collect_metadata = False
            
        base_path = os.path.split(path)[0]
        tfrecords_dir = os.path.join(base_path, 'tfrecords')
        if not os.path.exists(tfrecords_dir):
            os.makedirs(tfrecords_dir)
        return tfrecords_dir, path
        
    def _write_to_tfrecord(self, writer, sample_generator, samples_per_tfrecord):
        classes = set()
        for count, sample in enumerate(sample_generator):
            classes.update(sample['labels'])
            example = self._create_example(sample)
            writer.write(example.SerializeToString())
            if count == samples_per_tfrecord-1:
                break
        return classes

    def _create_example(self, example):
        image = self._load_image(example['path'])
        feature = {
            "bounding_boxes": self._float_feature_list(example["bounding_boxes"]),
            "labels": self._float_feature_list(example["labels"]),
            "image_path": self._image_path_feature(example["path"]),
            "image": self._image_feature(image),
            "image_shape": self._float_feature_list(image.get_shape())
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
        
    def _load_image(self, image_path):
        image = tf.io.decode_image(tf.io.read_file(image_path))
        return image
        
    def _image_path_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def _float_feature_list(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _image_feature(self, value):
        return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

    def get_dataset(self, preprocessor, mode='train', format='classification'):
        tfrecords_dir, _ = self._get_tfrecords_dir(mode)
        filenames = tf.io.gfile.glob(f"{tfrecords_dir}/*.tfrec")
        parse_tfrecord_fn = self._parse_tfrecord_fn
        preprocessing_fn = preprocessor.get_preprocessing_fn(mode, format)
        prepare_sample = self._prepare_sample
        
        dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
            .shuffle(self.dataset_size)
            .map(lambda x: parse_tfrecord_fn(x, format), num_parallel_calls=AUTOTUNE)
            .map(lambda x: prepare_sample(x, preprocessing_fn, format), num_parallel_calls=AUTOTUNE)
            .batch(self.batch_size)
        )
        if mode == 'validation':
            dataset = dataset
        elif mode == 'test':
            dataset = dataset
        return dataset
    
    def _parse_tfrecord_fn(self, example, format):
        if format == 'detection':
            return self._parse_detection_fn(example)
        elif format == 'classification':
            return self._parse_classification_fn(example)
            
    def _parse_detection_fn(self, example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_path": tf.io.FixedLenFeature([], tf.string),
            "bounding_boxes": tf.io.VarLenFeature(tf.float32),
            "labels": tf.io.VarLenFeature(tf.float32),
            "image_shape": tf.io.VarLenFeature(tf.float32)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        example["bounding_boxes"] = tf.sparse.to_dense(example["bounding_boxes"])
        example["labels"] = tf.sparse.to_dense(example["labels"])
        example["image_shape"] = tf.sparse.to_dense(example["image_shape"])
        return example
        
    def _parse_classification_fn(self, example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_path": tf.io.FixedLenFeature([], tf.string),
            "labels": tf.io.VarLenFeature(tf.float32),
            "image_shape": tf.io.VarLenFeature(tf.float32)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        example["labels"] = tf.sparse.to_dense(example["labels"])
        example["image_shape"] = tf.sparse.to_dense(example["image_shape"])
        return example
    
    def _prepare_sample(self, features, preprocessing_fn, format):
        if format == 'detection':
            return self._prepare_detection_sample(features, preprocessing_fn)
        elif format == 'classification':
            return self._prepare_classification_sample(features, preprocessing_fn)
            
    def _prepare_classification_sample(self, features, preprocessing_fn):
        features = preprocessing_fn(features)
        image = features['image']
        labels = features['labels']
        output = []
        for i in range(self.num_of_boxes):
            output.append(tf.one_hot(tf.cast(labels[i], tf.uint8), depth=self.num_classes))
        return image, tuple(output)
        
        
    def _prepare_detection_sample(self, features, preprocessing_fn):
        features = preprocessing_fn(features)
        bounding_boxes = tf.reshape(features['bounding_boxes'], [-1])
        image = features['image']
        labels = features['labels']
        output = []
        for i in range(self.num_of_boxes):
            value = tf.concat([tf.one_hot(tf.cast(labels[i], tf.uint8), depth=self.num_classes), bounding_boxes[4*i:4*i+4]], 0)
            output.append(value)
        return image, tuple(output)
        
    @property
    def dataset_size(self):
        return self._metadata['num_train_samples']
        
    def get_sample(self, dataset):
        iterator = dataset.__iter__()
        return next(iterator)

    def get_metadata(self):
        return self._metadata

    def update_metadata(self, dict_):
        self._metadata.update(dict_)
        
    @property
    def num_classes(self):
        classes = self._metadata.get('classes') #TODO: create dataset to collect data and use method properly
        return len(classes) 
        
    def class_mapping(self):
        if "class_mapping" not in self._metadata:
            dict_ = {}
            for i in self._metadata['classes']:
                if 0 < i <10:
                    dict_[i] = i
                elif i == 10:
                    dict_[i] = 0
                elif i == 0:
                    dict_[i] = 'None'
            self._metadata['class_mapping'] = dict_
        return self._metadata['class_mapping']
    
    def save_metadata(self):
        path = self._train_file_path
        base_path = os.path.split(path)[0]
        path = os.path.join(base_path,'metadata.pickle')
        with open(path, 'wb') as handle:
            pickle.dump(self._metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_metadata(self):
        path = self._train_file_path
        base_path = os.path.split(path)[0]
        path = os.path.join(base_path,'metadata.pickle')
        if os.path.isfile(path):
            with open(path, 'rb') as handle:
                self._metadata = pickle.load(handle)