import pandas as pd
import tensorflow as tf
RECOMENDATION_COUNT = 30

uRatings = pd.DataFrame.from_csv("../sheets/Processed/userratings.tsv")
titles = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep='\t', header=0)
genreList = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
             'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror',
             'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
             'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']

filename_queue = tf.train.string_input_producer(["MoviesML.tsv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])

  coord.request_stop()
  coord.join(threads)






# Available inputs:

    # Large movie dataset
        # id : IMDB i of movie in numerical form
        # startYear : year the movie was released
        # runtimeMinutes : (optional) how long the movie was
        # [genres] : (1,28) array of bits that denote what movie genres each movie belongs to
        # averageRating : the ratng of each movie in IMDB
        # [billing] : (optional)  an array of each important person that worked on the movie
    # User ratings
        # id : IMDB ID of each movie that hhas been rated
        # rating : user rating from 0 to 1 of each movie

# Desired outputs
    # array of about (1,30) movie IDs
    # (optional) array of about (1,30) confidence/similarity scores, each for a different movie
userRatings = tf.placeholder(tf.float32, shape=[None, 2])
id = tf.placeholder(tf.float32, shape=[None,1])
year = tf.placeholder(tf.float32, shape=[None,1])
# rating = tf.placeholder(tf.float32, shape=[None, 1])
genres = tf.placeholder(tf.float32, shape=[None, len(genreList)])
inputs = tf.concat([id, year, rating, genres], axis=1)

train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)


#hidden layers
h1 = tf.layers.dense(inputs, 30, activation="relu")
h2 = tf.layers.dense(h1, 5, activation="relu")

# nearest_neighbors=tf.Variable(tf.zeros([K]))
rawOutputs = tf.layers.dense(h2,1, activation=tf.nn.relu)

outputs = tf.cast(rawOutputs, tf.float32)


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(loss)


#initialize of all variables
init=tf.global_variables_initializer()

#start of tensor session
with tf.Session() as sess:
    sess.run(init)


    for index, row in
        sess.run(outputs, {id: titles["tconst"], year: titles["startYear",
                rating : titles["averageRating", genres: genres]})  # help
