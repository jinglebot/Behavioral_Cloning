# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2017_06_14_15_30_17_972.jpg "Model Visualization"
[image2]: ./examples/center_2017_06_22_07_34_49_960.jpg "Recovery Image"
[image3]: ./examples/center_2017_06_14_15_36_42_498.jpg "Recovery Image"
[image4]: ./examples/center_2017_06_22_07_56_49_305.jpg "Recovery Image"
[image5]: ./examples/center_2016_12_01_13_30_48_287_unflipped.jpg "Normal Image"
[image6]: ./examples/center_2016_12_01_13_30_48_287_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network
* [writeup_report.md](./writeup_report.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with:

* three 5x5 filter sizes and depths between 24 and 48 (model.py lines 114-116)
* two 3x3 filter sizes and depths of 64 (model.py lines 117-118)
* one flatten layer (model.py line 119)
* three fully connected layers and depths between 100 and 10 (model.py lines 120-122)
* the output layer (model.py line 123)

The model includes RELU layers to introduce nonlinearity (code line 114-118). Data is preprocessed in the generator with the process_image() function by cropping 70 px at the top and 25 px at the bottom, resizing to 64x64x3, changing randomly the image brightness and converting to RGB, and normalizing in batches (model.py lines 20-36).

I also added processing of the images in the 'drive.py' file (drive.py lines 67-75) to match the processing done in 'model.py'(model.py lines 20-36).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17, 131). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I first used the training sample data given by Udacity. Then, I tried my own acquired data which consists of a combination of two laps center lane driving, one lap recovering from the left and right sides of the road, one lap driving in reverse(clockwise direction), and a series of recoveries from the bridge, the dirtpath, the red-striped lane and the solid yellow-lined lane. I also tried different mixes of Udacity's and my own training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to

My first step was to use a convolution neural network model similar to the LeNet, 2 convolution layers and 2 fully connected layers. I thought this model might be appropriate because it's basically a simple network and it was  the first suggested in the Behavior Modelling overview video.

In order to gauge how well the model was working, I split my image and steering angle data into 80% training and 20% validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so as to get a decreasing MSE by adding a Dropout after the first layer, checking it for overfitting, add another to the second, checking again, until all the layers had a Dropout.

Then I tried MaxPooling first on each of the fully connected layers only, then on each of the convolutional layers until I got a good graph of the training and validation MSEs. Sad to say, I was getting good training values but not the validation values. So, I changed architecture and used a network similar to NVIDIA's, 5 convolution layers and 3 fully connected layers. Eventually, I was able to get decent values.

I tried to run the simulator to see how well the car was driving around track one. As expected it went off the road. There were a few spots where the vehicle fell off the track, the bridge, the dirtpath and the very curvy red-striped lane and yellow-lined lane. To improve the driving behavior in these cases, I augmented the training data and videoed a lot of additional images on those spots.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 114-123) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 64x64x3 RGB image   							|
| Convolution 24x5x5   	| 2x2 stride 									|
| RELU					|												|
| Convolution 36x5x5   	| 2x2 stride 									|
| RELU					|												|
| Convolution 48x5x5    | 2x2 stride 									|
| RELU					|												|
| Convolution 64x3x3    | 1x1 stride 									|
| RELU					|												|
| Convolution 64x3x3    | 1x1 stride 									|
| RELU					|												|
| Flattening			| 3x3x64 image, outputs 576						|
| Fully connected		| 576 array, outputs 100						|
| Fully connected		| 100 array, outputs 50							|
| Fully connected		| 50 array, outputs 10							|
| Fully connected		| 10 array, outputs 1							|
|						|												|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving, one lap of recoveries on certain spots from the left and right side of the road, one lap of driving clockwise. I included the center, left and right images of samples with angles greater than 0.5 and less than -0.5. For other samples, I would randomly include none, 1 or 2 images(left or right) but never the center image. Here is an example image of center lane driving:

![Center Lane Driving][image1]

I then made additional recording of the vehicle recovering from the left side and right sides of the road where it usually falters and back to center so that the vehicle would learn to get back on the center when it goes to the side of the road. These images show what a recovery looks like starting from ... :

![From the bridge's right side][image2]
![From the dirt path right side][image3]
![From the red-striped lane's right side][image4]

Then I repeated this process on the left side in order to get more data points.

To augment the dataset,  also flipped images and angles thinking that this would balance the data for both left and right sides.For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

I placed this part in a generator so that only the number of samples required per batch would be processed saving precious memory.
I had 40,000 number of data points and my batch size is 32. I then preprocessed this data still in the generator by cropping, resizing, randomly changing brightness and normalizing.

I randomly shuffle the data set everytime I run the generator and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used callbacks to save the best runs. The number of epochs is in 50 but the ideal number of epochs was 3 as evidenced by best models usually being reached on the third epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
