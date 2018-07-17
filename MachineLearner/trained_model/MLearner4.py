import tensorflow as tf
import numpy as np
# import tensorflow.contrib.eager as tfe
import os
# import tensorflow.contrib.eager as tfe
import os

import numpy as np
import tensorflow as tf

# tfe.enable_eager_execution()

#sess_cpu = tf.Session(config = tf.ConfigProto(device_count={'GPU': 0}))

#### https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/eager/custom_training_walkthrough.ipynb#scrollTo=tMAT4DcMPwI-

""" TODO
* Find out how to adapt the code to give us a predicted score instead of trying to fit into a category, ask alex
#eagar model tf checkpoints saving and loading
* DON Save and load the trained model, so we don't have to train it every time
* DONE Play around with the number of neurons and hidden layers
* DONE Find out wtf is happening with the loss function, as it always gives a huge number at the first epoch then 1.066 every subsequent epoch
* Manage to predict ratings based on different inputs
* DONE Implement a training dataset
* Make a for loop that would run code 30 or so times to give us a list of highly rated movies
* Take in the corrected ratings from the userratings.tsv file and use them to retrain the machine learner. Right now they're only trained 
    based on the IMDB ratings. Training multiple times using the same list of userratings would probably do the trick, but IDK
*Pass all the movies through the loss order in decending order
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
    ['/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesML.tsv'],
    batch_size,
    column_names=columns,
    label_name=label_name,
    field_delim="\t",
    shuffle=True,  # Shuffles data to make sure that the program doesn't take in movie id as a value relevant to the score
    num_epochs=1)

train_dataset = train_dataset.map(pack_features_vector)  # Runs the pack_features_vector function on the dataset

INPUT_SIZE = 28 + 3  # How many values are being passed as input

train_dataset = train_dataset.repeat()

# IDK what this does, and at this point I'm too afraid to find out
# Maybe sets the features and labels variables
# features, labels = next(iter(train_dataset))
next_training = train_dataset.make_one_shot_iterator().get_next()


#try to find how to use parallel models


# Creates a neural network model. First hidden layer has 17 neurons, the second 10 and it has 1 output
#
# model = tf.keras.Sequential([
#   #tf.keras.layers.Dense(796, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),  # input shape required
#   tf.keras.layers.Dense(800, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)), #under1.1 = 306 min = 1.88
#   #tf.keras.layers.Dense(850, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),#under1.1 = 231 min = 1.089
#   #tf.keras.layers.Dense(950, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),  # under1.1 = 16
#   #tf.keras.layers.Dense(900, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),  # under1.1 = 304 min 1.088
#   #tf.keras.layers.Dense(750, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)), TOO LOW
#   #tf.keras.layers.Dense(775, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)), TOO LOW
#   #tf.keras.layers.Dense(825, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),
#   #tf.keras.layers.Dense(200, activation=tf.nn.relu),
#   #tf.keras.layers.Dense(128, activation=tf.nn.tanh),
#   tf.keras.layers.Dense(1)  # 1 output neuron as rating
# ])

def Predict(model, input_features):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predictions = sess.run(model.prediction, {model.inputs: input_features})
        return predictions


class Model:
    def __init__(self, input_size=INPUT_SIZE, hidden_size = 800, rating_scale=10,
                 optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)):
        self.inputs = tf.placeholder(tf.float32, [None, input_size])
        self.hidden = tf.layers.dense(self.inputs, hidden_size, activation = tf.nn.tanh)
        self.prediction_raw = tf.layers.dense(self.hidden, 1)
        self.prediction = rating_scale * tf.nn.sigmoid(self.prediction_raw)
        self.prediction = tf.squeeze(self.prediction, axis=1)
        self.actual_rating = tf.placeholder(tf.float32, [None,])
        self.loss = tf.losses.absolute_difference(labels = self.actual_rating, predictions = self.prediction)
        self.train = optimizer.minimize(self.loss)



# # Predictions, may or may not be debug code
# predictions = model(features)
# print("    Predictions: {}".format(predictions))
#
# # IDK what this does, and at this point I'm too afraid to find out
# print("softmax: {}".format(tf.nn.softmax(predictions[:5])))
#
# print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
# print("    Labels: {}".format(labels))

# def loss(model, x, y):
#     y_ = model(x)
#     # Reshapes the tensor returned by the model from a (64, 1) tensor to an (n,) array, usually a (64,) tensor but n is set to
#     # be the shape length of y, to avoid runtime errors once the final batch is reached, as there won't be enough entries to fill
#     # up the (64, ) shape, but for example it would be a (23, ) or (51, ) tensor.
#     y_ = tf.reshape(y_, shape=(tf.shape(y)[0], ))
#
#     return tf.losses.absolute_difference(labels=y, predictions=y_)
#
# # defines l as the loss
# l = loss(model, features, labels)
# print("Loss test: {}".format(l))
#
# # No idea what the gradient is
# def grad(model, inputs, targets):
#     with tf.GradientTape() as tape:
#         loss_value = loss(model, inputs, targets)
#     return loss_value, tape.gradient(loss_value, model.trainable_variables)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)  # Optimizer
#
# global_step = tf.train.get_or_create_global_step() # dunno
#
# loss_value, grads = grad(model, features, labels) # double dunno

# print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
#                                           loss_value.numpy()))
#
# optimizer.apply_gradients(zip(grads, model.variables), global_step) # triple dunno
#
# print("Step: {},         Loss: {}".format(global_step.numpy(),
#                                           loss(model, features, labels).numpy()))

## Note: Rerunning this cell uses the same model variables
model = Model()
saver = tf.train.Saver()
folder_path = './trained_model/'


# keep results for plotting
train_loss_results = []



def Train2ElectricBoogaloo():
    train_loss_results = []
    ## train_accuracy_results = []
    num_epochs = 25000 + 1  # The amount of epochs the code will run for
    save_frequency = 50
    model = Model()
    saver = tf.train.Saver()
    folder_path = './trained_model/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with tf.Session() as sess:
        training_best = 0
        min_loss = 10
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):  # Training the model, will train for 300 epochs
            ## epoch_accuracy = tfe.metrics.Accuracy()
            # Get next batch of examples and labels
            features, labels = sess.run(next_training)
            # print('Features:', features.shape,'\n',features)
            # print('Features:', labels.shape,'\n',labels)
            # Training loop - using batches of 64
            # for x, y in train_dataset:
            #     # Optimize the model
            #     loss_value, grads = grad(model, x, y)
            #     optimizer.apply_gradients(zip(grads, model.variables),
            #                               global_step)
            #
            #     # Track progress
            #     epoch_loss_avg(loss_value)  # add current batch loss
            #     # compare predicted label to actual label
            #     ## epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.float32), y)
            feed_dict = {model.inputs: features, model.actual_rating: labels}
            sess.run(model.train, feed_dict)
            loss = sess.run(model.loss, feed_dict)
            # end epoch
            train_loss_results.append(loss)
            ## train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 1000 == 0:  # Print loss every 10 epochs
                print("Epoch {:03d}: Loss: {:.3f}".format(epoch, loss))
            if loss < 1.100:
                training_best += 1
            min_loss = min(min_loss, loss)

            if epoch % save_frequency == 0 and epoch != 0:
                saver.save(sess, folder_path + 'pg-checkpoint', epoch)

        print('Training finished.')
        saver.save(sess, folder_path + './trained_model/', epoch)

        print("Epoch beneath 1.1: " + str(training_best) + " Lowest Value: " + str(min_loss))



#predict_dataset = tf.contrib.data.make_csv_dataset()  # Load dataset from MoviesML.tsv
#     ['/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesMLShort3.tsv'],
#     batch_size,
#     column_names=columns,
#     label_name=label_name,
#     field_delim="\t",
#     shuffle=True,  # Shuffles data to make sure that the program doesn't take in movie id as a value relevant to the score
#     num_epochs=1)
#
# predict_dataset = predict_dataset.map(pack_features_vector)
#
# next_predicted = predict_dataset.make_one_shot_iterator().get_next()
# #print(next_predicted)
# #with tf.Session() as sess:
#     #features = sess.run(next_predicted)
# print(Predict(model, next_predicted[0]))
#
# # for line in predict_dataset:
# #     score = Predict(model, line)
# #     print(score)
#prediction_file = ["/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesMLShort3.tsv"]

path = '/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesMLShort3.tsv'


def predict(filepath):
    movie_list = []
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(folder_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        with open(filepath, "r") as f:
            first = True
            for line in f:
                if not first:
                    input_prediction = np.fromstring(line, dtype=float, sep='\t')
                    # print(Predict(model, np.reshape(input_prediction, (1, 31))))
                    titleid = "tt" + str(input_prediction[0])
                    reshaped_data = np.reshape(input_prediction, (1, 31))
                    prediction = Predict(model, reshaped_data)
                    prediction = [titleid, prediction]
                    print('Prediction:', prediction)
                    movie_list.append(prediction)
                else:
                    first = False
    return movie_list


predict(path)
# with tf.Session() as sess:
#     # check our folder for saved checlpoints
#     checkpoint = tf.train.get_checkpoint_state(folder_path)
#     # restore the checkpoint for our agent
#     saver.restore(sess, checkpoint.model_checkpoint_path)
#     # Run our agent through a series of testing episodes
#     for episode in range(testing_episodes):
#         state = env.reset()
#         episode_rewards = 0
#         for step in range(max_steps_per_episode):
#             env.render()
#             # Get Action
#             action_argmax = sess.run(my_agent.choice, feed_dict={my_agent.input: [state]})
#             action_choice = action_argmax[0]
#
#             state_next, reward, done, _ = env.step(action_choice)
#             state = state_next
#
#             episode_rewards += reward
#
#             if done or step + 1 >= max_steps_per_episode:
#                 print("Rewards for episode " + str(episode) + ": " + str(episode_rewards))
#                 break
# for x in train_dataset:
#     y = model(x)
#     movie_list.append(x[0])
# movie_list = sorted(movie_list, reverse = True)
# print(movie_list)



# Plot out the loss
# fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
# fig.suptitle('Training Metrics')
#
# axes[1].set_ylabel("Loss", fontsize=14)
# axes[1].set_xlabel("Epoch", fontsize=14)
# axes[1].plot(train_loss_results)