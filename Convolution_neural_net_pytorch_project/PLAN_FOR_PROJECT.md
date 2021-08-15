# **Deep into CNN**

END-EVAL REPORT

### **Week 1**

- **Plan/Goals**    

  - **Numerical data** : Multi layer Perceptron ( MLP ) :
    - Regression python Implementation
    - Gradient Descent,relu layer, MSE loss
    - Binary Classification,sigmoid layer,BCE loss
    - Multiclass Classification,softmax layer

  - **NLL loss MLP + PyTorch** :
    - Linear Algebra, Single Layer NN, Training, - Inference and Validation : Illustrated Through Pytorch
    - Implement 1-hidden layer NN using PyTorch but train in python

- **Tasks**

  - **Content reading on Regression And Shallow NN Using Python**.

    - **Things Learnt:**

      - **Cost/Loss function:** Structure and the basic use of cost/loss functions were taught

      - **(Stochastic) Gradient descent:** It includes the process of linear descent and how it works. It also included the limit and use of parameters like learnrate while implementing gradient descent

      - **Python Implementation from scratch** : All the above things were taught in python without using any big library. It was done to give a clear picture of how different things/functions work and implemented directly.

  - **Completing the programming exercises shared and updating github repo with practice code and completed exercises.**

    - **Things learnt:**

      - **Basic data handling with numpy and pandas** : Mentees were taught how to load data from csv and clean the data with the help of numpy and pandas. Other than the basic functions, mentees were also taught One-hot encoding of the data, normalisations and its importance,and some implementations of matplotlib to view data.

      - **Implementing sigmoid and error calculation functions** : sigmoid and error calculation functions were taught and implemented with numpy.

      - **Training and implementing shallow NN from scratch in python:** After all the functions were made, mentees implemented the them along with error term calculation and back- propagation to make a shallow NN and there accuracy and loss were observed after every epochs

### **Week 2**

- **Plans/Goals**

  - **Intro to CNN :**
    - Simple Feed-forward Network :
      - Flatten images first and then treat them as numerical data.

    - Convolutional Neural Networks :
      - Use Spatial Information

    - Compare results with MLP on MNIST data
    - Start Using PyTorch.

- **Tasks**

  - **Content reading on Neural Networks in pytorch**

    - **Things learnt -**

      - **Backprop:** Though it was already implemented in week 1, but it was used only in shallow NN and therefore it was included again in week 2 to give a complete sense of its implementation on deep NN. Also, this time it was implemented using pytorch.

      - **Softmax** : Mentees were taught about the use, importance and implementation of softmax functions from scratch and in pytorch.

      - **Basics of Pytorch** : Assignments were given to teach mentees the implementation of different functions (related to NN) in pytorch.

  - **Complete the programming exercises shared and update github repo with practice code and solved assignments:** Total of 8 assignments were given. Mentees were taught how to -

      - Load and handle data using pytorch
      - Uses and implementation of dataloaders and importance of parameters like batch-size
      - Loading already available data using torch vision and the process of normalisation
      - Basic structure and documentation about the NN in pytorch
      - Criterions like CrossEntropyLoss, NLLLoss,
      - Optimizer like Adam, SGD
      - Use, importance and complete implementation of pytorch autograd and its use with loss functions.
      - Train neural networks and setting of hyperparameters and learning rate to adjust data along with RELu and LogSoftmax functions
      - Importance of Validating the data during training and its implementation using the Dropout function.
      - Saving and loading back the already trained models and its importance
      - An optional exercise to make a Cat-Dog identifier was also given along with the basic framework required to make it.

  - **Hackathon-1 starts** : The Hackathon 1 data set was very noisy and was given to give an idea of how Kaggle and Hackathon works. It also signified an important fact that simple fully connected NN can sometimes be inefficient to train and predict the data accurately.

### **Week 3**

- **Plans/Goals**
  - **Layers:** Maxpool, Average and Dropout ,Fully connected layers in combination with Convolutional layers.
  - **LeNet:**
    - Convolution
    - Pooling
    - Fully connected layers

  - Competition of Hackathon 1

  - Practice assignments on CNN

- **Tasks**

  - **Hackathon-1 submission** : The hackathon was an open one. Hosted on kaggle and included a total of over 800 teams ([Tabular Playground Series - Jun 2021](https://www.kaggle.com/c/tabular-playground-series-jun-2021)).

  - **Hackathon-2 starts:** It is based on training NN and making predictions for RGB images.

  - **Read a famous shared paper and update the github repo with practice code and solved implementation of that paper.** Mentees were asked to choose one of the following SOTA models on ImageNET classification papers and implement it.

    - AlexNet
    - VGG
    - Inception
    - Xception
  - **Things learnt:**

- **Filters** : Creating filters(edge detection, sobel,customised) and applying them to images.
- **Visualize Convolution** : Visualize four filtered outputs (a.k.a. activation maps) of a convolutional layer.
- **MNIST** : Train an MLP to classify images from the [MNIST database](http://yann.lecun.com/exdb/mnist/) hand-written digit database.
- **Different types of layers** : Train a CNN to classify images from the CIFAR-10 database and define a CNN architecture using Convolutional layers and Max Pooling layers.
- **Hyperparameter tuning** : Deciding a loss and optimization function that is best suited for this classification task.
- **LeNet-5** : Implement a modified LeNet-5 for MNIST digits classification but use max instead of average pooling and use fully connected instead of gaussian final layer.

### **Week 4 - Week5**

- **Plans/Goals**

  - **Optimizer variation:**
    - SGD with Momentum, Nesterov and Adam

  - **Overfitting and Regularization**
    - L1, L2
    - Batch-Norm

  - **Hyperparameter tuning**
    - Variable learning rate
    - Weight Initialization : Xavier, He Normal

- **Tasks**

  - **Paper 1 Submission** : Out of the four given papers on SOTA models ( VGG, Inception, Exception. AlexNet), mentees were asked to choose one and implement it.

    - **Things Learnt:**
      - **The thinking process:** Mentees learnt about the process of &quot;deciding&quot; of where and how the layers should be added to improve the efficiency/accuracy of the model.

      - **New Ideas:** The different and innovative ways to implement an idea in a model, for example - the splitting of layers of data and making them pass through different processing layers and combining them again in the end, was never taught to mentees before.

      - **The practical use of Optimizers and Batch-normalization:** Mentees were able to see the use of optimizers, non-linearity along with batch-normalization (that was taught this week) in action through these papers

  - **Hackathon-2 ends:** Hackathon was based on the following data-set-

    - [https://www.kaggle.com/gpiosenka/100-bird-species](https://www.kaggle.com/gpiosenka/100-bird-species)

It was a data set consisting of RGB images of birds divided into 275 species.

The Mentees were advised to use their SOTA models on this data set along with the optimisation techniques taught in this week.

- -
    - **Things Learnt:**

        - **Data modification:** Though the data was already clean, mentees were still supposed to apply transformation to it to make it more general- a step to prevent overfitting.

        - **Practice** on implementing the SOTA models on a data set that requires a deep model and is difficult to get good accuracy on.

        - **Self-Implementation of optimization techniques:** The mentees were able to use the techniques to find the best parameters to fit the data-set. They were also encouraged to slightly modify the SOTA model to introduce dropouts and other things that they thought fit.

### **Week 5 - Week 6**

- **Plans/Goals:**

  - **Autoencoders**

    - Convert High dimension to Low dimension data
    - Should be able to convert Low to high with minimum error

    - MLP
      - First flatten images i.e. convert to numerical data
      - As a (ineffective) compression method

    - Convolution
      - For Denoising images
      - Uses Transposed Convolutions

  - **Generative Adversarial networks**

    - Generate new data points as efficiently possible Generator
    - Generate fake data Discriminator
    - Recognize fake data and penalize generator
    - Generator and Discriminators Compete with Each Other !!!

- **Tasks:** Due to the time constraints ( partially due to the break for Y20 End- Sem and partially due to the Y19 Internships), the week 5 - week 6 plans were cut short and only the following task was given -

  - **Complete the programming exercises shared and update the github repo with practice code and solved assignments-** there were a total of 3 assignments. Mentees were taught the following things -

    - Simple encoding and decoding (reproducing) of data using a neural network.

    - Repeating the above process by adding convolution instead of simple neural network and observing improvement in results

    - Adding a lot of noise in the input data and making the model deeper with the goal of de-noising the data, by setting the clean data as the target of noisy data for the model.

