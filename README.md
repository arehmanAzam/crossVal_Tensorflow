# crossVal_Tensorflow
This repository contains code for training training deep learning model using k-cross validation for image data.


## Prerequisites
We expect to have proir knowledge of keras and tensorflow. Also see keras documentation for [Sequential model](https://keras.io/getting-started/sequential-model-guide) and [Compilation](https://keras.io/getting-started/sequential-model-guide/#compilation)


You can use any version of the packages but we have used the following versions: 

Packages      | Versions
------------- | -------------
Keras         | 2.1.3
Tensorflow    | 1.8.0
Numpy         | 15.4.0
Matplotlib    | 3.0.2

## Prior Work
We assume that your dataset is in the form of images. These frames are located in every class-folder of training. 
Before getting to this point of work, your code might look like this 

```

# Load entire dataset
path_dataset_train='/home/activity/train'

# Design model
model.Sequential()
model.add(Conv2D(filters, kernel_size, strides=(1, 1))
.....
# Compile model with your optimizers
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
## Using training_crossVal Function
In python headers where packages are imported code this line
```
from cross_Validation import training_crossVal

```
then pass following parameters for training of the model initialized above.
```
dataset_train='/home/abdulrehman/images/train'

training_crossVal(kvalidation_splits=7,train_batch_size=2100,model_train=model,epochs=15,image_directory_path=dataset_train):
```
###Note
On every epoch the dataset fed to the model will be train_batch_size/kvalidation_splits.
So if you have total training dataset of 10000 images, train_batch_size=2100, kvalidation_splits=7. 
In every epoch, generator will pick 2100 images with labels randomly and 7 sub-epochs will run having batch of 300 for training and 300 for validation.

So ####total_epochs run in the above example will be 7x15=105 epochs

The epochs are great in number but the batch size taken for each epoch becomes small and in each epoch data is rotated randomly. It enhances the training capability and time for training is same for the model if other generator with no-sub epochs are used.


## Authors
* [Abdulrehman Azam](https://github.com/arehmanAzam)

## License
This project is under GNU General Public License v3.0 see [License](https://github.com/arehmanAzam/3D-CNN_DataGenerator/blob/master/LICENSE) file

