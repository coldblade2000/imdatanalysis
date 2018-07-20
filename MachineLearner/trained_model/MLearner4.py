# import tensorflow.contrib.eager as tfe
# import tensorflow.contrib.eager as tfe
import os

import numpy as np
import tensorflow as tf

import Recommender.Recommender as rec

from difflib import get_close_matches

import pandas as pd

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
* DONE Make a for loop that would run code 30 or so times to give us a list of highly rated movies
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
# train_dataset = tf.contrib.data.make_csv_dataset(  # Load dataset from MoviesML.tsv
#     ['/home/student/PycharmProjects/imdatanalysis/sheets/Processed/usertitleratingsML.tsv'],
#     batch_size,
#     column_names=columns,
#     label_name=label_name,
#     field_delim="\t",
#     shuffle=True,  # Shuffles data to make sure that the program doesn't take in movie id as a value relevant to the score
#     num_epochs=1)

train_dataset = train_dataset.map(pack_features_vector)  # Runs the pack_features_vector function on the dataset

INPUT_SIZE = 28 + 3  # How many values are being passed as input

train_dataset = train_dataset.repeat()

# IDK what this does, and at this point I'm too afraid to find out
# Maybe sets the features and labels variables
# features, labels = next(iter(train_dataset))
next_training = train_dataset.make_one_shot_iterator().get_next()

path = '/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesMLShort3.tsv'

#try to find how to use parallel models


# Creates a neural network model. First hidden layer has 17 neurons, the second 10 and it has 1 output
#

#   #tf.keras.layers.Dense(950, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),  # under1.1 = 16
#   #tf.keras.layers.Dense(900, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),  # under1.1 = 304 min 1.088
#   #tf.keras.layers.Dense(750, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)), TOO LOW
#   #tf.keras.layers.Dense(775, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)), TOO LOW
#   #tf.keras.layers.Dense(825, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),
#   #tf.keras.layers.Dense(200, activation=tf.nn.relu),
#   #tf.keras.layers.Dense(128, activation=tf.nn.tanh),
#   tf.keras.layers.Dense(1)  # 1 output neuron as rating

# model = tf.keras.Sequential([
#   #tf.keras.layers.Dense(796, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),  # input shape required
#   tf.keras.layers.Dense(800, activation=tf.nn.relu, input_shape=(INPUT_SIZE,)), #under1.1 = 306 min = 1.88
#   #tf.keras.layers.Dense(850, activation=tf.nn.tanh, input_shape=(INPUT_SIZE,)),#under1.1 = 231 min = 1.089
# ])

class Model:
    def __init__(self, input_size=INPUT_SIZE, hidden_size = 800, rating_scale=10,
                 optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)):
        #These are the inputs that have been provided by Diego
        self.inputs = tf.placeholder(tf.float32, [None, input_size])
        #We have one hidden layer with 800 neurons
        self.hidden = tf.layers.dense(self.inputs, hidden_size, activation=tf.nn.tanh)
        #This condenses the hidden layer into a single layer
        self.prediction_raw = tf.layers.dense(self.hidden, 1)
        #This scales the number into an actual prediction on a scale of 1 to 10
        self.prediction = rating_scale * tf.nn.sigmoid(self.prediction_raw)
        #This makes sure that the shape of the data is compatable with the AI
        self.prediction = tf.squeeze(self.prediction, axis=1)
        #This is the tru+e IMDB rating
        self.actual_rating = tf.placeholder(tf.float32, [None,])
        #This is the loss which shows how incorrect the machine learner is
        self.loss = tf.losses.absolute_difference(labels = self.actual_rating, predictions = self.prediction)
        #This is where we actually train the machine Ito become smarter
        self.train = optimizer.minimize(self.loss)
        for var in vars(self):
            if isinstance(var, tf.Tensor):
                print(var)


    #Just like the name states, this is how the machine predicts what movies the user will like
def Predict(model, input_features):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predictions = sess.run(model.prediction, {model.inputs: input_features})
        return predictions

def Train2ElectricBoogaloo(load=True, folder_path = './final_movie_app/', num_epochs = 10000 + 1):
    train_loss_results = []
    ## train_accuracy_results = []
    save_frequency = 50
    folder_path = os.path.abspath(folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model = Model()
    saver = tf.train.Saver()
    model_path = folder_path + '/pg-checkpoint'
    with tf.Session() as sess:
        training_best = 0
        min_loss = 10
        try:
            if not load:
                raise Exception('Do not load')
            checkpoint = tf.train.get_checkpoint_state(folder_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('Model restored from saved training data.')
        except:
            sess.run(tf.global_variables_initializer())
            print('Initialized new model')

        for epoch in range(num_epochs):  # Training the model, will train for 80000 epochs
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

            if epoch % 1000 == 0:  # Print loss every 1000 epochs
                print("Epoch {:03d}: Loss: {:.3f}".format(epoch, loss))
            if loss < 1.100:
                training_best += 1
            min_loss = min(min_loss, loss)

            if epoch % save_frequency == 0 and epoch != 0:
                saver.save(sess, model_path, epoch)

        print('Training finished.')
        saver.save(sess, model_path, epoch)
        print('Model was restored from saved training data.')
        print('The model data was saved to "{}".'.format(model_path))
        print("The lowest loss value was: " + str(min_loss))


def predictFromFile(filepath):
    movie_list = []
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(folder_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        with open(filepath, "r") as f:
            first = True
            for line in f:
                if not first:
                    input_prediction = np.fromstring(line, dtype=float, sep='\t')
                    titleid = rec.getFullid("tt", input_prediction[0])
                    reshaped_data = np.reshape(input_prediction, (1, 31))
                    prediction = Predict(model, reshaped_data)
                    prediction = [titleid, prediction]
                    print('Prediction:', prediction)
                    movie_list.append(prediction)
                else:
                    first = False
    return movie_list


def predict(input_prediction):
    # (1,31) matrix
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(folder_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        reshaped_data = np.reshape(input_prediction, (1, 31))
        prediction = Predict(model, reshaped_data)
    return prediction[0]


# predictFromFile(path)
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


#Find where I am saving and loading the model. It cannot find the right path YET
def suggest_titles(folder_path = './final_movie_app/', number_recomendations = 10, movies_csv = '/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesML.tsv', batch_size=10000):
    movies_csv = '/home/student/PycharmProjects/imdatanalysis/sheets/Processed/MoviesML.tsv'
    movies = pd.read_csv(movies_csv, sep="\t")
    movies.drop("averageRating", axis=1, inplace=True)
    folder_path = os.path.abspath(folder_path)
    with tf.Session() as sess:
        recomendations = []
        model = Model()
        saver = tf.train.Saver()
        load = True
        # sess.run(tf.global_variables_initializer())
        try:
            if not load:
                raise Exception('Do not load')
            checkpoint = tf.train.get_checkpoint_state(folder_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('Model restored from saved training data.')
        except Exception as e:
            print('Error:\n',e)
            print('Initialized new model')
        examples = []
        items = list(movies.iterrows())
        for index, row in items:
            row = row.tolist()
            examples.append(row)
            if len(examples) >= batch_size or index == items[-1][0]:
                feed = {
                  model.inputs: examples
                } #Curly braces = dictionary
                predicted_rating = sess.run(model.prediction, feed)
                suggestions = [(examples[i][0], predicted_rating[i]) for i in range(len(examples))]
                recomendations += suggestions
                if len(recomendations) > number_recomendations:
                    recomendations = sorted(recomendations,key=lambda suggestion: -suggestion[1],
                                            )[: number_recomendations]
                # print('Current predictions:')
                examples = []
                # for index, suggestion in enumerate(recomendations):
                #     print(index,':',suggestion)
        return recomendations

#True and false to vary between the training and preditions
train = False
if train:
    Train2ElectricBoogaloo(num_epochs=3000)
else:
    suggestions = suggest_titles()
    print('Suggested title id:')
    for sug in suggestions:
        id, rating = sug
        print('Title id: ', id, '\t\tRating: ', rating)
