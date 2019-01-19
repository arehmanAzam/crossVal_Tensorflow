import numpy as np
import cv2
import os
import tensorflow as tf
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy  import array,random
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from time import time
from dataset import load_cached



class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


csv_logger = CSVLogger('log_autoencoder1.csv',
                            append=True, separator=',')
#  tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1,
#                          write_graph=True, write_images=False)
tensorboard =TrainValTensorBoard(write_graph=False)
#
callbacks_list2 = [csv_logger,tensorboard]

global image_paths_train
global labels_train
def random_batch(train_batch_size=2100):


    # Number of images (transfer-values) in the training-set.
    num_images = len(image_paths_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = array(image_paths_train)[idx]
    y_batch = array(labels_train)[idx]

    return x_batch, y_batch

def training_crossVal(kvalidation_splits=7,train_batch_size=2100,model_train=None,epochs=15,image_directory_path=None):
    counter_epoch = 0
    dataset = load_cached(cache_path='my_dataset_cache_repo.pkl',
                          in_dir=image_directory_path)
    image_paths_train, cls_train, labels_train = dataset.get_training_set()
    for i in range(epochs):
        x, y = random_batch(train_batch_size=train_batch_size)
        images_split = np.split(x, kvalidation_splits)
        images_labels = np.split(y, kvalidation_splits)
        del x
        # for image_paths in images_split:

        for count_i in range(kvalidation_splits):
            image_paths = images_split[count_i]
            train_image_label = images_labels[count_i]
            train_image = np.empty((len(image_paths), 224, 224, 3))
            # train_image=np.array([[]])
            for i in range(len(image_paths)):
                image = cv2.imread(image_paths[i])
                if (image is not None):
                    resized_image = cv2.resize(image, dsize=(224, 224))
                    # resized_image_float=im2double(resized_image)
                    np_image = np.reshape(resized_image, (224, 224, 3))
                    np_image = np_image.astype('float32')
                    train_image[i] = np_image
                else:
                    np.delete(image_paths, (i), axis=0)
                    np.delete(train_image, (i), axis=0)
                    np.delete(train_image_label, (i), axis=0)
                # train_image=np.append(train_image,np_image)
            del image, resized_image, np_image, image_paths

            if count_i == kvalidation_splits - 1:
                image_paths_val = images_split[0]
                val_image_label = images_labels[0]
                val_image = np.empty((len(image_paths_val), 224, 224, 3))
            else:
                image_paths_val = images_split[count_i + 1]
                y_paths_val = images_labels[count_i + 1]
                val_image = np.empty((len(image_paths_val), 224, 224, 3))

            for j in range(len(image_paths_val)):
                image = cv2.imread(image_paths_val[j])
                if (image is not None):
                    resized_image = cv2.resize(image, dsize=(224, 224))
                    # resized_image_float=im2double(resized_image)
                    np_image = np.reshape(resized_image, (224, 224, 3))
                    np_image = np_image.astype('float32')
                    val_image[j] = np_image
                else:
                    np.delete(image_paths_val, (j), axis=0)
                    np.delete(val_image, (j), axis=0)
                    np.delete(y_paths_val, (i), axis=0)
            # np.save('images.npy',train_image_np)
            del image, resized_image, np_image, image_paths_val
            model_train.fit(x=train_image, y=train_image_label, epochs=1, batch_size=1, callbacks=callbacks_list2,
                            validation_data=(val_image, y_paths_val))
            counter_epoch = counter_epoch + 1
            print('Total epochs=' + str(counter_epoch))
            model_train.save('logs/autoencoder.h5')
            del train_image
            del val_image
        del images_split
    return model_train