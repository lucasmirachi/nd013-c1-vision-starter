**TF Records** are TensorFlow's custom data format. Even though they are not technically required to train a model with TensorFlow, they can be very useful. For some pre-existing TensorFlow APIs, such as the object detection API that we will use for the final project, a TF Record format is required to train models.

Summarizing:

* TF records are TensorFlow custom data format
* They are not required to train a model with Tensorflow, but may help to speed up data loading
* Their structure is defined by proto files
* Tf records are created using Protocol buffers (protobuf), a mechanism to serialize data


## Objective

The goal of this exercise is to make you familiar with the tf record format. In particular, your job is to convert the data from the Waymo Open Dataset into the tf record format used by the Tensorflow Object Detection API. As a Machine Learning Engineer, you will often have to convert dataset from one format to another and this is a great example of such task.

## Details

You can read more about the Waymo Open Dataset data format [here](https://waymo.com/open/data/perception/). Each tf record files contains the data for an entire trip made by the car, meaning that it contains images from the different cameras as well as LIDAR data. Because we want to keep our dataset small, we are implementing the <mark>create_tf_example</mark> function to create cleaned tf records files.

**NOTE** - We are using the Waymo Open Dataset github repository to parse the raw tf record files. We would recommend to follow [this tutorial](https://github.com/waymo-research/waymo-open-dataset) to better understand the data format before diving into this exercise. 

## Tips

This [document](https://github.com/Jossome/Waymo-open-dataset-document) provides an overview of the dataset structure.

Later on, we will leverage the Tensorflow Object Detection API to train Object Detection models. In the API tutorial, you can find an example of <mark>create_tf_example</mark> [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records).

### Aditional Resources

* [Using TFRecord and tf.train.Example from the TensorFlow documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord)