import ast
import json

import numpy as np
import pandas as pd

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
    movies = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
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
        count = 0
        for line in openfile:
            count = count + 1
            if not first:
                person = []
                tSplit = line.split("\t")
                for i, x in enumerate(tSplit):
                    person.append(x)
                if person[0] == currentId:
                    personList.append(person)
                elif currentId == '':
                    currentId = person[0]
                    personList.append(person)
                else:
                    try:
                        titleDF.at[currentId, 'billing'] = json.dumps(personList)
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
    # titleDF = titleDF.Index.isin(movies['index'])
    pd.DataFrame.to_csv(titleDF, '../sheets/Processed/MoviesBilling2.tsv', sep='\t')

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

def createListFromString(string):
    outstr = str(string)
    outstr = outstr.replace("\n","")
    outstr = outstr.replace("[ ","[")
    outstr = outstr.replace(".  ",",")
    outstr = outstr.replace(".","")
    array = ast.literal_eval(outstr)
    return array

def addBinaryGenresProperly():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesTrunc.tsv", sep="\t", header=0)
    print('Finished loading')
    series = pd.DataFrame(titleDF['genres'].map(createListFromString),columns=genreList)
    series.rename(lambda x:genreList[x])
    print("starting export")
    print(titleDF.tail(8))
    titleDF.to_csv("../sheets/Processed/MoviesML5.tsv", sep="\t")
    print(titleDF.head(8))

def addBinaryGenresIter():
    with open("../sheets/Processed/MoviesTrunc.tsv") as openfile:
        first = True
        titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesTrunc.tsv", sep="\t", header=0)
        for itr in genreList:
            titleDF[itr] = 0
        print(titleDF.head(3))
        for idx, line in enumerate(openfile.readlines()):
            if not first:
                tSplit = line.split("\t")
                array = getBinaryArray(tSplit[3])
                for idx2, genre in enumerate(array):
                    if genre == 1:
                        titleDF.ix[idx-1,genreList[idx2]] = genre
            else:
                first = False
        titleDF.to_csv("../sheets/Processed/MoviesAttempt3.tsv", sep="\t")


def convertSlowly():
    with open("../sheets/Processed/MoviesAttempt.tsv", "w") as writefile, open("../sheets/Processed/MoviesML.tsv", "r") as openfile:
        first = True # bool to avoid iterating through the header of principals.tsv
        for line in openfile.readlines():
            if not first:
                tSplit = line.split("\t")
                print(len(tSplit))
                tSplit[3] = tSplit[3].replace("[ ", "").replace(".  ", "\t").replace(".]", "")
                writeline = ""
                for i in tSplit:
                    writeline = writeline + i +"\t"
                writefile.write(writeline)
                # person = []
                # tSplit = line.split("\t")
                # for i, x in enumerate(tSplit):
                #     person.append(x)  # Creates a dictionary for each cast/crew member.
                # if person[0] == currentId:
                #     personList.append(person)
                # elif currentId == '':
                #     currentId = person[0]
                #     personList.append(person)
                # else:
                #     try:
                #         titleDF.at[currentId, 'billing'] = personList
                #         print(titleDF.at[person[0], 'billing'], " : ", currentId)
                #         personList = [person]
                #         currentId = person[0]
                #     except KeyError:
                #         print("Error with ", currentId)
                # print(person)
            else:
                linestr = str(line)
                new = ""
                for x in genreList:
                    new = new + x + "\t"
                new = new[:-2]
                writefile.write(linestr.replace("genres", new))
                first = False
            print("finished loading, now exporting sheet")
        openfile.close()
        writefile.flush()
        writefile.close()

def isolateMovies():
    titles = pd.DataFrame.from_csv("../sheets/Processed/MoviesBilling2.tsv", sep="\t", header=0)
    titles = titles[titles.endYear.str.contains("N") == True]
    titles = titles[titles.seasonNumber.isnull()]
    titles.to_csv("../sheets/Processed/MoviesBilling2Trunc.tsv", sep="\t")

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

def fixGenres():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    for idx, genre in enumerate(genreList):
        # titleDF.genres.str[2:-2].
        titleDF[genre] = np.nan
        titleDF[genre] = titleDF[genre].str

def numerateTConst():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesBilling2Trunc.tsv", sep="\t")
    print('Finished loading')
    titleDF.reset_index(inplace=True)
    print(titleDF.head(3))

    # Add empty 'billing' column to spreadsheets
    titleDF['realid'] = titleDF['tconst']
    titleDF.tconst = titleDF.tconst.str[2:]
    titleDF.set_index("tconst", inplace=True)
    titleDF.to_csv("../sheets/Processed/MoviesBilling2Trunc.tsv", sep="\t")
    print(titleDF.head(3))

def dropExtraColumns():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    billing = titleDF[['billing']].copy()
    print('Finished loading')
    titleDF.reset_index(inplace=True)
    print(titleDF.head(3))
    # Add empty 'billing' column to spreadsheets
    titleDF.drop("genres", axis=1, inplace=True)
    titleDF.drop("billing", axis=1, inplace=True)
    input("Everything alright?")
    print("saving")
    titleDF.to_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    print("saved moviesml")
    billing.to_csv("../sheets/Processed/MoviesBilling.tsv", sep="\t")
    print(titleDF.head(3))

def fixruntimeMinutes():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    titleDF.set_index("tconst", inplace=True)
    titleDF.replace(str(titleDF.at[4,"runtimeMinutes"]), "",inplace=True)
    titleDF["runtimeMinutes"] = pd.to_numeric(titleDF['runtimeMinutes'], errors='coerce')

    titleDF.to_csv("../sheets/Processed/MoviesML.tsv", sep="\t")

def dropEmptyRatings():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    titleDF = titleDF[titleDF['runtimeMinutes'].notnull()]
    titleDF.to_csv("../sheets/Processed/MoviesML.tsv", sep="\t")

def onlyrand5000():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesML.tsv", sep="\t")
    titleDF = titleDF.sample(n=5000,)
    titleDF.to_csv("../sheets/Processed/MoviesMLShort2.tsv", sep="\t")


def trimMoviesBilling2():
    titleDF = pd.DataFrame.from_csv("../sheets/Processed/MoviesBilling2.tsv", sep="\t")
    titleDF = titleDF
    titleDF.drop("genres", axis=1, inplace=True)
    titleDF.drop("originalTitle", axis=1, inplace=True)
    titleDF.drop("runtimeMinutes", axis=1, inplace=True)

# titleDF= pd.DataFrame.from_csv("../sheets/Processed/TitlesFull.tsv")
numerateTConst()  # Add empty 'billing' column to spreadsheet
# titleDF['billing'] = np.nan
# titleDF['billing'] = titleDF['billing'].astype(object)

# open principals.tsv to append its contents to titles.tsv
