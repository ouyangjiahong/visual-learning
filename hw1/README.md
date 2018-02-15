# Assignment 1: Object Classification with TensorFlow!

- [Visual Learning and Recognition (16-824) Spring 2018](https://sites.google.com/andrew.cmu.edu/16824-spring2018/home)
- Created By: [Rohit Girdhar](http://rohitgirdhar.github.io)
- TAs: [Lerrel Pinto](http://www.cs.cmu.edu/~lerrelp/), [Senthil Purushwalkam](http://www.cs.cmu.edu/~spurushw/), [Nadine Chang](https://www.ri.cmu.edu/ri-people/nai-chen-chang/) and [Rohit Girdhar](http://rohitgirdhar.github.io)
- Please post questions, if any, on the piazza for HW1.
- Total points: 100

In this assignment, we will learn to train multi-label image classification models using the [TensorFlow](www.tensorflow.org) (TF) framework. We will classify images images from the PASCAL 2007 dataset into the objects present in the image. Your task in this assignment is to fill in the parts of code, as described in this document, perform all experiments, and submit a report with your results and analyses. You are free to use any TensorFlow built-in high-level APIs, such as `tf.contrib.slim` or `tf.contrib.keras`, as long as you can follow the code structure we define in the steps of this assignment. Feel free to google how to do certain things if you get stuck, but put proper attribution. It is *not* acceptable to google "alexnet for PASCAL classification in tensorflow" and copy-paste that code, as that would probably not follow the code structure we define in the assignment.

In all the following tasks, coding and analysis, please write a short summary of what you tried, what worked (or didn't), and what you learned, in the report. Write the code into the files as specified. Submit a zip file (`ANDREWID.zip`) with all the code files, and a single `REPORT.pdf`, which should have commands that TAs can run to re-produce your results/visualizations etc. Also mention any collaborators or other sources used for different parts of the assignment.

## Software setup

If you are using AWS instance setup using the provided instructions, you should already have most of the requirements installed on your machine. In any case, you would need the following python libraries installed:

1. TensorFlow (1.3+)
2. Numpy
3. Pillow (PIL)
4. sklearn (v0.19)

## Task 0: MNIST 10-digit classification in TensorFlow (5 points)

MNIST is a dataset containing handwritten digits from 0-9, formatted as 28x28px monochrome images. It has been popularly used for debugging and experimenting with convolutional neural networks (CNNs). In this task, we will start with understanding the basic workings of TensorFlow utilities provided for building CNNs. We will follow [MNIST Layers](https://www.tensorflow.org/tutorials/layers) official tutorial. If you have trouble understanding this tutorial, or want more background, you can look at [MNIST](https://www.tensorflow.org/get_started/mnist/beginners) and [Deep MNIST](https://www.tensorflow.org/get_started/mnist/pros) tutorial. It is also recommended you go through the [Estimator](https://www.tensorflow.org/extend/estimators) tutorial, which is the new TF high-level API.

For simplicity, I already provide the code from the MNIST tutorial in [`00_mnist.py`](00_mnist.py). Try running that code using `python 00_mnist.py`. It will start printing the loss per-iterations. After 20000 iterations, it will run the trained model on test data and print the classification accuracy. Go through the code and make sure you understand the different parts of it.

#### Q 0.1: What test accuracy does your model get?

#### Q 0.2: What happens if you train for more iterations (30000)?

#### Q 0.3: Make the training loss and validation accuracy curve. Show for at least 100 points between 0 to 20000 iterations.
*Hint: You will need to run the `mnist_classifier.train` in a loop; train for a few iterations, run the evaluate to get the current accuracy, and so on.*

Yay! We can now read numbers from images :-) This technology, developed by LeCun and collaborators in 1990s became the basis of automated check and zip code processing.

## Task 1: 2-layer network for PASCAL multi-label classification (20 points)

Numbers are easy. Lets try to recognize some natural images.
We start by modifying this code to read images from the PASCAL 2007 dataset. Following steps will guide you through the process.

### Data setup

We first need to download the image dataset and annotations. Use the following commands to setup the data, and lets say it is stored at location `$DATA_DIR`.

```bash
$ # First, cd to a location where you want to store ~0.5GB of data.
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ tar xf VOCtrainval_06-Nov-2007.tar
$ # Also download the test data
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
$ cd VOCdevkit/VOC2007/
$ export DATA_DIR=$(pwd)
```


The first step is to write a data loader to load this PASCAL data. Since there are only about 10K images in this dataset, we can simply load all the images into CPU memory, along with the labels. The important thing to note is that PASCAL can have multiple objects present in the same image. Hence, this is a multi-label classification problem, and will have to be tackled slightly differently.

We provide some starter code for this task in `01_pascal.py` by slightly modifying our MNIST codebase. You need to fill in some of the functions, as outlined next.

#### Q 1.1: Write a data loader for PASCAL 2007.
Find the function definition for `load_pascal`. As the function docstring says, the function takes as input the `$DATA_DIR` and the split (`train`/`val`/`trainval`/`test`), and outputs all the images, labels and weights from the dataset. For `N` images in the split, the images should be `np.ndarray` of shape `NxHxWx3`, and labels/weights should be `Nx20`. The labels should be 1s for each object that is present in the image, and weights should be 1 for each label in the image, except those labeled as ambiguous. All other values should be 0. For simplicity, resize all images to a canonical size (eg, 256x256px).
*Hint: The dataset contain a `ImageSets/Main/` folder, with files named `<class_name>_<split_name>.txt`. Use those files to find images that are in the different splits of the data. Look at the README to understand the structure and labeling.*

Since the data is in `numpy` format, we use the `tf.estimator.inputs.numpy_input_fn` data loader, that takes care of constructing batches, shuffling etc. This is already provided in [`01_pascal.py`](01_pascal.py).

#### Q 1.2: Modify the MNIST model function to be suitable for multi-label classification.
Next we need to write the model function for PASCAL. I provide an empty function definition, but feel free to copy over from your MNIST code. We will be using the same model from MNIST (bad idea, I know, but lets give it a shot). We need to take care of a couple of things:

1. Now that our images are much larger (256px vs 28px before), you might need to change the size in the reshape layer before the fully connected (`dense`) layer.
2. You will no longer need the `tf.one_hot` function as you are already producing a one-hot-ish representation from `load_pascal`.
3. Perhaps most importantly, we need to change the final non-linearity. A standard solution for multi-label classification problems is to consider each class as a separate binary classification problem, and to use `tf.sigmoid` as the activation, and `tf.losses.sigmoid_cross_entropy` as the loss function. Use these to replace the `softmax` loss and activation we used for MNIST.


With all this code in place, we should now be able to train the model. For now we will use the same training parameters as we used for MNIST. The other thing to figure out is the evaluation. A standard metric for multi-label evaluation is [mean average precision (mAP)](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). I already provide the code for evaluation; just make sure your `model_fn` can return an `EstimatorSpec` for the predict mode (it should return the probability for each class).

#### Q 1.3: Same as before, show the training and accuracy (mAP) curves. Train for only 1000 iterations.


## Task 2: Lets go deeper! AlexNet for PASCAL classification (20 points)

As you might have seen, the performance of our 2-layer model that worked perfectly on MNIST, was pretty low for PASCAL. This is expected as PASCAL is much more complex than MNIST, and we need a much beefier model to handle it. Copy over your code from `01_pascal.py` to `02_pascal_alexnet.py`, and lets implement a deep CNN.

In this task we will be constructing a variant of the [alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) architecture, known as CaffeNet. If you are familiar with Caffe, a prototxt of the network is available [here](https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/train_val.prototxt).

### Network

Here is the exact model we want to build. I use the following operator notation for the architecture:

1. Convolution: A convolution with kernel size `k`, stride `s`, output channels `n`, padding `p`, is represented as `conv(k, s, n, p)`.
2. Max Pooling: A max pool operation with kernel size `k`, stride `s` as `max_pool(k, s)`.
3. Fully connected: For `n` units, `fully_connected(n)`.

```txt
ARCHITECTURE:
	-> image
	-> conv(11, 4, 96, 'VALID')
	-> relu()
	-> max_pool(3, 2)
	-> conv(5, 1, 256, 'SAME')
	-> relu()
	-> max_pool(3, 2)
	-> conv(3, 1, 384, 'SAME')
	-> relu()
	-> conv(3, 1, 384, 'SAME')
	-> conv(3, 1, 256, 'SAME')
	-> max_pool(3, 2)
	-> flatten()
	-> fully_connected(4096)
	-> relu()
	-> dropout(0.5)
	-> fully_connected(4096)
	-> relu()
	-> dropout(0.5)
	-> fully_connected(20)
```

#### Q 2.1: Replace the MNIST model we were using before with this model.

### Solver parameters

We would also need to modify the solver settings from what we used on MNIST. 

1. Change the optimizer to SGD + Momentum, with momentum of 0.9.
2. Initialize the conv and FC weights using a `gaussian(0, 0.01)` initializer, and biases using a `zeros` initializer. You may refer to the Caffe prototxt above for exact details.
3. Use a exponentially decaying learning rate schedule, that starts at 0.001, and decays by 0.5 every 10K iterations. You should train it for at least 40K iterations, which should take half hour on the AWS nodes.

#### Q 2.2: Implement the above solver parameters. This should require small changes to our previous code.

### Data Augmentation

Since we are training a model from scratch on this small dataset, it is important to perform some basic data augmentation to avoid overfitting. Add random crops and left-right flips when training, and do a center crop when testing. *Hint: Note that you can use ops such as `tf.image.random_flip_left_right`, `tf.random_crop` etc directly into your `cnn_model_fn` function, and do not need to perform this augmentation manually in the data loader. This is one of the strengths of TF, it allows you to specify all data pre-processing as a part of the computation graph and can optimally schedule all operations.*


#### Q 2.3: Implement the data augmentation and generate the loss and mAP curves as before.

*Hint: You may refer to a previous work, Kra½henu¿½hl et al. (ICLR'16), for more tips on setting hyper-parameters for this task. Feel free to explore slight modifications of the architecture.*

## Task 3: Even deeper! VGG-16 for PASCAL classification (15 points)

Hopefully we all got much better accuracy with the deeper model! Since 2012, many other deeper architectures have been proposed, and [VGG-16](https://arxiv.org/abs/1409.1556) is one of the popular ones. In this task, we attempt to further improve the performance with the "very deep" VGG-16 architecture.

#### Q 3.1: Modify the network architecture from Task 2 to implement the VGG-16 architecture (refer to the original paper). Use the same hyperparameter settings from Task 2, and try to train the model. Add the train/test loss/accuracy curves into the report.

### Setting up tensorboard
TensorFlow ships with an awesome visualization tool called TensorBoard. It can be used to visualize training losses, network weights and other parameters. Now that we're training much deeper network, lets hook up tensorboard to get better understanding of these networks.

If you have been using the `Estimator` API, then your code is already storing logs for tensorboard in the `models_dir`! You can visualize it by running `tensorboard --logdir $MODEL_DIR --port 6006`, and view the UI using a browser, at `<aws-public-ip>:6006`. Try that out.

#### Q 3.2: The task in this section is to log the following entities: a) Training loss, b) Learning rate, c) Histograms of gradients, d) Training images and e) Network graph into tensorboard. Add screenshots from your tensorboard into the report.

## Task 4: Standing on the shoulder of the giants: finetuning from ImageNet (20 points)
As we have already seen, deep networks can sometimes be hard to optimize, while other times lead to heavy overfitting on small training sets. Many approaches have been proposed to counter this, eg, [Krahenbuhl et al. (ICLR'16)](http://arxiv.org/pdf/1511.06856.pdf) and other works we have seen in un-/self-supervised learning. However, the most effective approach remains pre-training the network on large, well-labeled datasets such as ImageNet. While training on the full ImageNet data is beyond the scope of this assignment, people have already trained many popular/standard models and released them online. In this task, we will initialize the VGG model from the previous task with pre-trained ImageNet weights, and *finetune* the network for PASCAL classification. You can download the pre-trained VGG-16 model for TensorFlow from [here](goo.gl/BakqKs), and rename as `./vgg_16.ckpt`.

You might want to look at `tf.train.SessionRunHook` to load models into `Estimator`s. Also, the pre-trained model used the following namescope hierarchy:
```txt
vgg_16/
	conv1/
		conv1_1/
			weights
			biases
		conv1_2/
		... and so on
```

It would help if you define your VGG model using a similar naming hierarchy (look up `tf.variable_scope`), as that would help you load the model without converting blob names from the ones in the model to ones in your code.

#### Q 4.1: Load the pre-trained weights upto fc7 layer, and initialize fc8 weights and biases from scratch. Then train the network as before and report the training/validation curves and final performance on PASCAL test set.
Use similar hyper-parameter setup as in the scratch case, however, use 1/10th the learning rate, number of iterations and learning rate step size.

## Task 5: Analysis (20 points)

By now we should have a good idea of training networks from scratch or from pre-trained model, and the relative performance in either scenarios. Needless to say, the performance of these models is way stronger than previous non-deep architectures we used until 2012. However, final performance is not the only metric we care about. It is important to get some intuition of what these models are really learning. Lets try some standard techniques.

### Conv-1 filters
Extract and compare the conv1 filters from CaffeNet in Task 2, at different stages of the training. Show at least 3 data points.

### Nearest neighbors
Pick 10 images from PASCAL test set from different classes, and compute the nearest neighbors of those images over the test set. You should use and compare the following feature representations for the nearest neighbors:

1. pool5 features from the AlexNet (trained from scratch)
2. fc7 features from the AlexNet (trained from scratch)
3. pool5 features from the VGG (finetuned from ImageNet)
4. fc7 features from VGG (finetuned from ImageNet)

### tSNE visualization of intermediate features
We can also visualize how the feature representations specialize for different classes. Take 1000 random images from the test set of PASCAL, and extract `fc7` features from those images. Compute a 2D [tSNE projection](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) of the features, and plot them with each feature color coded by the GT class of the corresponding image. If multiple objects are active in that image, compute the color as the "mean" color of the different classes active in that image. Legend the graph with the colors for each object class.

### Are some classes harder?
Show the per-class performance of your scratch (alexnet) and pre-trained (VGG) models. Try to explain, by observing examples from the dataset, why some classes are harder or easier than other (consider the easiest and hardest class). Do some classes see large gains due to pre-training? Can you explain why that might happen?

## Task 6 (Extra Credit): Improve the classification performance (20 points)
Many techniques have been proposed in the literature to improve classification performance for deep networks. In this section, we try to use a recently proposed technique called [*mixup*](https://arxiv.org/abs/1710.09412). The main idea is to augment the training set with linear combinations of images and labels. Read through the paper and modify your model to implement mixup. Report your performance, along with training/test curves, and comparison with baseline in the report.


## Acknowledgements
Parts of the starter code are taken from official TensorFlow tutorials. Many thanks to the original authors!
