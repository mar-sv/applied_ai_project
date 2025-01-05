import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
import math
import time
import tensorflow as tf

# Load the MNIST dataset


def get_minst_data(train_size, test_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reduce the size of the dataset (e.g., 10,000 training samples and 2,000 test samples)
    train_size = 5000  # Number of training samples to use
    test_size = 1000    # Number of test samples to use

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = x_test[:test_size]
    y_test = y_test[:test_size]

    # Normalize data (0 and 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten
    x_train_flattened = x_train.reshape(x_train.shape[0], -1)
    x_test_flattened = x_test.reshape(x_test.shape[0], -1)

    return (x_train_flattened, y_train), (x_test_flattened, y_test)


def getEuclideanDistance(single_point, array):
    nrows, ncols, nfeatures = array.shape[0], array.shape[1], array.shape[2]
    points = array.reshape((nrows*ncols, nfeatures))

    dist = (points - single_point)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    dist = dist.reshape((nrows, ncols))
    return dist


def get_accuracy(confusion_matrix):

    total_correct = np.trace(confusion_matrix)   # sum of diagonal elements
    total_samples = np.sum(confusion_matrix)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy


def SOM_Test(trainingData, som_, classes, grid_, ConfusionMatrix, ndim=60):
    nfeatures = trainingData.shape[1]
    ntrainingvectors = trainingData.shape[0]

    nrows = ndim
    ncols = ndim

    nclasses = np.max(classes)

    som_cl = np.zeros((ndim, ndim, nclasses+1))

    for ntraining in range(ntrainingvectors):
        trainingVector = trainingData[ntraining, :]
        class_of_sample = classes[ntraining]
        # Compute the Euclidean distance between the training vector and
        # each neuron in the SOM map
        dist = getEuclideanDistance(trainingVector, som_)

        # Find 2D coordinates of the Best Matching Unit (bmu)
        bmurow, bmucol = np.unravel_index(
            np.argmin(dist, axis=None), dist.shape)

        som_cl[bmurow, bmucol, class_of_sample] = som_cl[bmurow,
                                                         bmucol, class_of_sample]+1

    for i in range(nrows):
        for j in range(ncols):
            grid_[i, j] = np.argmax(som_cl[i, j, :])

    for ntraining in range(ntrainingvectors):
        trainingVector = trainingData[ntraining, :]
        class_of_sample = classes[ntraining]
        # Compute the Euclidean distance between the training vector and
        # each neuron in the SOM map
        dist = getEuclideanDistance(trainingVector, som_)

        # Find 2D coordinates of the Best Matching Unit (bmu)
        bmurow, bmucol = np.unravel_index(
            np.argmin(dist, axis=None), dist.shape)

        predicted = np.argmax(som_cl[bmurow, bmucol, :])
        ConfusionMatrix[class_of_sample-1, predicted -
                        1] = ConfusionMatrix[class_of_sample-1, predicted-1]+1

    return grid_, ConfusionMatrix


def get_empty_confusionmatrix(classes):
    nclasses = np.max(classes)
    return np.zeros((nclasses, nclasses))


def get_grid(ndim):
    nrows = ndim
    ncols = ndim
    grid_color = np.zeros((nrows, ncols))
    return grid_color

# def display_weights()


def SOM(dispRes, trainingData, ndim=10, nepochs=10, eta0=0.1, etadecay=0.05, sgm0=20, sgmdecay=0.05, showMode=0, show_middle_weights=False, print_epoch=False):
    nfeatures = trainingData.shape[1]
    ntrainingvectors = trainingData.shape[0]

    nrows = ndim
    ncols = ndim

    mu, sigma = 0, 0.1
    np.random.seed(int(time.time()))
    som = np.random.normal(mu, sigma, (nrows, ncols, nfeatures))

    if showMode >= 1:
        print(f"SOM features before training: {ndim} \n")

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

        for k in range(nrows):
            for l in range(ncols):
                A = som[k, l, :].reshape((dispRes[0], dispRes[1]))
                ax[k, l].imshow(A, cmap="plasma")
                ax[k, l].set_yticks([])
                ax[k, l].set_xticks([])

    # Generate coordinate system
    plt.show()
    x, y = np.meshgrid(range(ncols), range(nrows))

    for t in range(1, nepochs+1):
        # Compute the learning rate for the current epoch
        eta = eta0 * math.exp(-t*etadecay)

        # Compute the variance of the Gaussian (Neighbourhood) function for the ucrrent epoch
        sgm = sgm0 * math.exp(-t*sgmdecay)

        # Consider the width of the Gaussian function as 3 sigma
        width = math.ceil(sgm*3)

        for ntraining in range(ntrainingvectors):
            trainingVector = trainingData[ntraining, :]

            # Compute the Euclidean distance between the training vector and
            # each neuron in the SOM map
            dist = getEuclideanDistance(trainingVector, som)

            # Find 2D coordinates of the Best Matching Unit (bmu)
            bmurow, bmucol = np.unravel_index(
                np.argmin(dist, axis=None), dist.shape)

            # Generate a Gaussian function centered on the location of the bmu
            g = np.exp(-((np.power(x - bmucol, 2)) +
                       (np.power(y - bmurow, 2))) / (2*sgm*sgm))

            # Determine the boundary of the local neighbourhood
            fromrow = max(0, bmurow - width)
            torow = min(bmurow + width, nrows)
            fromcol = max(0, bmucol - width)
            tocol = min(bmucol + width, ncols)

            # Get the neighbouring neurons and determine the size of the neighbourhood
            neighbourNeurons = som[fromrow:torow, fromcol:tocol, :]
            sz = neighbourNeurons.shape

            # Transform the training vector and the Gaussian function into
            # multi-dimensional to facilitate the computation of the neuron weights update
            T = np.matlib.repmat(
                trainingVector, sz[0]*sz[1], 1).reshape((sz[0], sz[1], nfeatures))
            G = np.dstack([g[fromrow:torow, fromcol:tocol]]*nfeatures)

            # Update the weights of the neurons that are in the neighbourhood of the bmu
            neighbourNeurons = neighbourNeurons + \
                eta * G * (T - neighbourNeurons)

            # Put the new weights of the BMU neighbouring neurons back to the
            # entire SOM map
            som[fromrow:torow, fromcol:tocol, :] = neighbourNeurons

        if t == nepochs/2 and show_middle_weights:
            print(f"SOM features middle of training: {ndim} \n")

            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

            for k in range(nrows):
                for l in range(ncols):
                    A = som[k, l, :].reshape((dispRes[0], dispRes[1]))
                    ax[k, l].imshow(A, cmap="plasma")
                    ax[k, l].set_yticks([])
                ax[k, l].set_xticks([])
            plt.show()
        if print_epoch:
            print(f"Epoch: {t}")

    if showMode >= 1:
        print(f"SOM features AFTER training: {ndim} \n")

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

        for k in range(nrows):
            for l in range(ncols):
                A = som[k, l, :].reshape((dispRes[0], dispRes[1]))
                ax[k, l].imshow(A, cmap="plasma")
                ax[k, l].set_yticks([])
            ax[k, l].set_xticks([])
        plt.show()
    return som
