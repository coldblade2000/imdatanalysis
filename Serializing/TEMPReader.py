import json
import numpy as np
import timeit
import pandas as pd

id = "tt8630480"
# count = 28
genreList = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']


def processSheets():
    titlebasics = pd.DataFrame.from_csv("../sheets/titlebasics.tsv", sep="\t", header=0)
    ratings = pd.DataFrame.from_csv("../sheets/ratings.tsv", sep="\t", header=0)
    crew = pd.DataFrame.from_csv("../sheets/crew.tsv", sep="\t", header=0)
    episode = pd.DataFrame.from_csv("../sheets/episode.tsv", sep="\t", header=0)
    result = titlebasics.join(ratings)
    result = result.join(crew)
    result = result.join(episode)
    result.drop("titleType", axis=1, inplace=True)
    print(result.tail(3))
    pd.DataFrame.to_csv(result, "../sheets/Processed/Titles.tsv", sep="\t")


def genres():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/Titles.tsv", sep="\t")
    rawGenres = titleDF.genres.unique()
    genres = []
    for genre in rawGenres:
        genre = str(genre).split(",")
        for individualGenre in genre:
            if individualGenre not in genres:
                genres.append(individualGenre)
    print(genres)
    return genres


# genres()
# titleDF = pd.DataFrame.from_csv("../sheets/Processed/Titles.tsv", sep="\t")


def principals():
    # load titles spreadsheet
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/Titles.tsv", sep="\t")
    print('Finished loading')

    # Add empty 'billing' column to spreadsheets
    titleDF['billing'] = np.nan
    titleDF['billing'] = titleDF['billing'].astype(object)

    # open principals.tsv to append its contents to titles.tsv
    with open("../sheets/principals.tsv") as openfile:
        first = True # bool to avoid iterating through the header of principals.tsv
        # keys = ["tconst", "ordering", "nconst", "category", "job", "characters"]
        currentId = ''
        personList = []
        for line in openfile:
            if not first:
                person = []
                tSplit = line.split("\t")
                for i, x in enumerate(tSplit):
                    person.append(x)  # Creates a dictionary for each cast/crew member.
                if person[0] == currentId:
                    personList.append(person)
                elif currentId == '':
                    currentId = person[0]
                    personList.append(person)
                else:
                    try:
                        titleDF.at[currentId, 'billing'] = personList
                        print(titleDF.at[person[0], 'billing'], " : ", currentId)
                        personList = [person]
                        currentId = person[0]
                    except KeyError:
                        print("Error with ", currentId)
                print(person)
            else:
                first = False
            print("finished loading, now exporting sheet")
        openfile.close()
    pd.DataFrame.to_csv(titleDF, '../sheets/Processed/TItlesidk.tsv', sep='\t')

def getBinaryArray(rawGen):
    list = str(rawGen).split(",")
    array = np.zeros(28)
    for genre in list:
        for i,item in enumerate(genreList):
            if genre == item:
                array[i] = 1
    return array



def addBinaryGenres():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesTrunc.tsv", sep="\t", header=0)
    titleDF['genres'] = titleDF['genres'].astype(object)
    print('Finished loading')
    titleDF['genres'] = titleDF['genres'].map(getBinaryArray)
    print("starting export")
    print(titleDF.tail(8))
    titleDF.to_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    print(titleDF.head(8))

def isolateMovies():
    titles = pd.DataFrame.from_csv("../sheets/Processed/Movies.tsv",sep="\t", header=0)
    titles = titles[titles.endYear.str.contains("N") == True]
    #titles = titles[titles.seasonNumber.isnull() == True]
    titles.to_csv("../sheets/Processed/Movies2.tsv", sep="\t")

def truncSheets():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesFull.tsv", sep="\t")
    print("Loaded")
    titleDF.drop("primaryTitle", axis=1, inplace=True)
    titleDF.drop("originalTitle", axis=1, inplace=True)
    titleDF.drop("isAdult", axis=1, inplace=True)
    titleDF.drop("endYear", axis=1, inplace=True)
    titleDF.drop("numVotes", axis=1, inplace=True)
    titleDF.drop("directors", axis=1, inplace=True)
    titleDF.drop("writers", axis=1, inplace=True)
    titleDF.drop("parentTconst", axis=1, inplace=True)
    titleDF.drop("seasonNumber", axis=1, inplace=True)
    titleDF.drop("episodeNumber", axis=1, inplace=True)
    print("Dropped columns")
    titleDF.to_csv("../sheets/Processed/MoviesTrunc.tsv", sep="\t")
    titleDF.head(3)


# titleDF= pd.DataFrame.from_csv("../sheets/Processed/TitlesFull.tsv")
addBinaryGenres()
# Add empty 'billing' column to spreadsheets
# titleDF['billing'] = np.nan
# titleDF['billing'] = titleDF['billing'].astype(object)

# open principals.tsv to append its contents to titles.tsv
