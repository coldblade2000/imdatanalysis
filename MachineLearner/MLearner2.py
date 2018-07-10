import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

tf.enable_eager_execution()

#### https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/eager/custom_training_walkthrough.ipynb#scrollTo=tMAT4DcMPwI-

""" TODO
* Find out how to adapt the code to give us a predicted score instead of trying to fit into a category, ask alex
* Save and load the trained model, so we don't have to train it every time
* Play around with the number of neurons and hidden layers
* Find out wtf is happening with the loss function, as it always gives a huge number at the first epoch then 1.066 every subsequent epoch
* Manage to predict ratings based on different inputs
* Implement a training dataset
* Make a for loop that would run code 30 or so times to give us a list of highly rated movies
* Take in the corrected ratings from the userratings.tsv file and use them to retrain the machine learner. Right now they're only trained 
    based on the IMDB ratings. Training multiple times using the same list of userratings would probably do the trick, but IDK
"""

# A list of every genre in alphabetical order

genreList = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
             'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
             'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
             'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']

# A list of every column in the dataset file. Adds the genrelist at the end
columns = ['tconst', 'startYear', "runtimeMinutes", "averageRating"] + genreList


def pack_features_vector(features, labels):
    # Pack the features into a single array.
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

# The names of all the different inputs, just the columns array except for "averageRating"
feature_names = columns[:3]+columns[4:]
# The name of the output/label. JUst "averageRating"
label_name = columns[3]

batch_size = 64  # The size of batches of movies that will be given to the machine learner at a time

train_dataset = tf.contrib.data.make_csv_dataset(  # Load dataset from MoviesML.tsv
    ['../sheets/Processed/MoviesML.tsv'],
    batch_size,
    column_names=columns,
    label_name=label_name,
    field_delim="\t",
    shuffle=True,  # Shuffles data to make sure that the program doesn't take in movie id as a value relevant to the score
    num_epochs=1)

train_dataset = train_dataset.map(pack_features_vector)  # Runs the pack_features_vector function on the dataset

INPUT_SIZE = 28 + 3  # How many values are being passed as input

# IDK what this does, and at this point I'm too afraid to find out
# Maybe sets the features and labels variables
features, labels = next(iter(train_dataset))

# Creates a neural network model. First hidden layer has 17 neurons, the second 10 and it has 1 output
model = tf.keras.Sequential([
  tf.keras.layers.Dense(17, activation=tf.nn.relu, input_shape=(INPUT_SIZE,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(1)
])

# Predictions, may or may not be debug code
predictions = model(features)
print("    Predictions: {}".format(predictions))

# IDK what this does, and at this point I'm too afraid to find out
print("softmax: {}".format(tf.nn.softmax(predictions[:5])))

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

def loss(model, x, y):
    y_ = model(x)
    # Reshapes the tensor returned by the model from a (64, 1) tensor to an (n,) array, usually a (64,) tensor but n is set to
    # be the shape length of y, to avoid runtime errors once the final batch is reached, as there won't be enough entries to fill
    # up the (64, ) shape, but for example it would be a (23, ) or (51, ) tensor.
    y_ = tf.reshape(y_, shape=(tf.shape(y)[0],))

    return tf.losses.absolute_difference(labels=y, predictions=y_)


# defines l as the loss
l = loss(model, features, labels)
print("Loss test: {}".format(l))

# No idea what the gradient is
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # Optimizer

global_step = tf.train.get_or_create_global_step() # dunno

loss_value, grads = grad(model, features, labels) # double dunno

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.variables), global_step) # triple dunno

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))

## Note: Rerunning this cell uses the same model variables

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 300 + 1  # The amount of epochs the code will run for

for epoch in range(num_epochs): # Training the model, will train for 300 epochs
    epoch_loss_avg = tfe.metrics.Mean()
    ## epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 64
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step)

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        ## epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.float32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    ## train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 10 == 0:  # Print loss every 10 epochs
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch,epoch_loss_avg.result()))


# Plot out the loss
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[1].set_ylabel("Loss", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_loss_results)


