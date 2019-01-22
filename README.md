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
from DataGenerator import training_crossVal

```
then pass path of the dataset folders to make or load cache_files for future usage

```
dataset_train=load_cached(cache_path='give path for making or loading .pkl file ',in_dir=path_dataset_train)
dataset_val=load_cached(cache_path='give path for making or loading .pkl file ',in_dir=path_dataset_val)
class_names=dataset_train.class_names
```
verify it by 

```
print(class_names)
```

Now get images path and labels using 

```
image_path_train,cls_train,labels_train=dataset_train.get_training_set()
image_path_val, cls_val,labels_val=dataset_val.get_training_set()
num_classes= dataset_train.num_classes
```

Concatenate both paths with parameters

```
partition={'train': image_path_train, 'validation': image_path_val}
```

Create and change parameters according to your needs for this 3D generators.

```
number_of_batches=10
frame_width=400
frame_height=256
frames_chunk=18
params={'dim': (frames_chunk,frame_width,frame_height),
        'n_channels': 3,
        'real_batchsize_custom': number_of_batches,
        'batch_size': frames_chunk*number_of_batches,
        'n_classes': num_classes,
        'shuffle': False
        'frames_chunk' : frames_chunk
        }
```

Get labels from the findLabels function

```
labels_training=findLabels(partition['train'], labels_train)
labels_validation=findLabels(partition['validation'], labels_val)
```


Pass the above created parameters to the main DataGenerator Class

```
training_generator = DataGenerator(partition['train'], labels_training, **params)
validation_generator = DataGenerator(partition['validation'], labels_validation, **params)
```

Give generators to your model.fit_generator() function for training the model

```
model.fit_generator(...
        generator=training_generator,
        validation_data=validation_generator,
        ..
        )
```
## Authors
* [Gulraiz Khan](https://github.com/gulraizk94) (Setting-up base of this project and generator with one batch)
* [Abdulrehman Azam](https://github.com/arehmanAzam) (Upgraded the generator for multiple batches)

## License
This project is under GNU General Public License v3.0 see [License](https://github.com/arehmanAzam/3D-CNN_DataGenerator/blob/master/LICENSE) file

