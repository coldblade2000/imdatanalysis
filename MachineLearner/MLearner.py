import pandas as pd
import tensorflow as tf
RECOMENDATION_COUNT = 30

uRatings = pd.DataFrame.from_csv("../sheets/Processed/userratings.tsv")
titles = pd.DataFrame.from_csv("../sheets/Processed/TitlesFull.tsv", sep='\t', header=0)
genreList = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']

uRatingsTensor = tf.convert_to_tensor(uRatings)
id = tf.placeholder(tf.float32, shape=[None,1])
year = tf.placeholder(tf.float32, shape=[None,1])
rating = tf.placeholder(tf.float32, shape=[None, 1])
genres = tf.placeholder(tf.float32, shape=[None, len(genreList)])
inputs = tf.concat([id, year, rating, genres], axis=1)

#hidden layers
h1 = tf.layers.dense(inputs, 30, activation="relu")
h2 = tf.layers.dense(h1, 5, activation="relu")

#K-near
# K=3 #how many neighbors

# nearest_neighbors=tf.Variable(tf.zeros([K]))
rawOutputs = tf.layers.dense(h2,RECOMENDATION_COUNT, activation=tf.nn.relu)

outputs = tf.cast(rawOutputs, tf.int32)

loss = tf.add(tf.squared_difference())

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(loss)

#setting accuracy as 0
accuracy=0

#initialize of all variables
init=tf.global_variables_initializer()

#start of tensor session
with tf.Session() as sess:
    sess.run(init)


    for index, row in
        sess.run(outputs, {id: titles["tconst"], year: titles["startYear",
                rating : titles["averageRating", genres: genres]})
