# Assignment 3: Action Classification + Fundamentals

- [Visual Learning and Recognition (16-824) Spring 2018](https://sites.google.com/andrew.cmu.edu/16824-spring2018/home)
- Created By: [Lerrel Pinto](http://www.cs.cmu.edu/~lerrelp)
- TAs: [Lerrel Pinto](http://www.cs.cmu.edu/~lerrelp/), [Senthil Purushwalkam](http://www.cs.cmu.edu/~spurushw/), [Nadine Chang](https://www.ri.cmu.edu/ri-people/nai-chen-chang/) and [Rohit Girdhar](http://rohitgirdhar.github.io)
- Submission Date: 18th April 2018 (Before Midnight)
- Please post questions, if any, on the piazza for HW3.
- Total points: 100

In the first part of this assignment, we will train action classification models on the [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). Since dealing with videos is time consuming, we provide ResNet18 features of images in the videos. In the second part of this assignment, we will answer some fundamenal questions regarding visual learning and recognition.

## Software setup

You can solve this assignment with any Python based deep learning package that you like. The upside is that you have complete freedom in designing the code. The downside is that we will be providing no code. You can complete this assignment on your local machine as well.

## Part 1: Action Classification

To learn more about action classification and the [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#introduction) dataset, check out the [website](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/). We have extracted ResNet18 features for you here: ([TrainSet](https://www.dropbox.com/s/y23pdfngf7uu4xn/annotated_train_set.p?dl=0),[TestSet](https://www.dropbox.com/s/2zc1vystx0161cr/randomized_annotated_test_set_no_name_no_num.p?dl=0)). The total size is less than 250MB.

### Part 1.0: Loading the data (5 Points)

Some free points to load this dataset. The dataset is in a single [pickle](https://wiki.python.org/moin/UsingPickle) file which can be easily loaded in Python. When you load the dataset using pickle, you will notice that it is a [dictionary](https://www.python-course.eu/dictionaries.php) with a single key: 'data'. The value object is a list of datapoints. Each datapoint is a dictionary that contains the 'features' of 10 frames of the video. In the TrainSet, you can see the 'class\_num' and 'class\_name' keys which you can use for training. In the TestSet, you do not have access to the class information. For the remaining subparts you will have to report the predicted classes on the TestSet and we will evaluate your models based on your predictions.

### Part 1.1 Training a simple classification network (20 Points)

The simplest way to do action classification is to classify individual frames and then pool the results of individual frames of video to get the final video classification. You will need to submit a link to a single text file 'part1.1.txt' where each row has the predicted label for the TestSet.

The first 5 entries, if the predicted labels are (43,21,0,50,19) for the first 5 elements of the TestSet, should look something like this:
```
43
21
0
50
19
```
Please ensure that the number of rows in the text file should be equal to the number of datapoints in the TestSet. 

Apart from the text file, it is suggested to include details of your model, training framework and how you evaluated the learned model, in your report.

### Part 1.2 Training a RNN network (25 Points)

Train a recurrent neural network that can get better performance than the simple classification network. Include the link to the text file 'part1.2.txt' that describes the predicted labels on the TestSet. Also include details of your model, training framework and how you evaluated your learned model, in your report.

## Part 2: Q&A (50 Points)

1. Why does using a regression loss for surface normal estimation lead to blurry results?

2. What is Batch Normalization and why is it useful? How does Batch Normalization vary for training and testing phases?

3. In a two stream network, when should you have shared weights and when separate?

4. You are training a two stream network for RGBD data (one stream for RGB and the other for D). Your dataset consists of 10,000 images for recognition. How would you initialize the network? How will you avoid overfitting in the network?

5. How would you convert a regression problem to a classification problem?

6. In a multi label image classification what loss would you use?

7. Why are CNNs used more for computer vision tasks than other tasks?

8. How do you find the best hyperparameters like learning rate, momentum, type of optimizer etc. for a new task at hand?

