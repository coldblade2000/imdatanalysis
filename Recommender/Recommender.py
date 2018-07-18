import json

import pandas as pd

ratings = pd.DataFrame.from_csv("~/PycharmProjects/imdatanalysis/sheets/userratings.tsv", sep="\t")
ratings = ratings.loc[ratings['type'] == 'nm']


def movieslookup(billing):
    # titleRatings = ratings.loc[ratings['type'] == 'tt']
    # titleBilling = billing[billing.index.isin(titleRatings.index)]
    # titleDF = titleDF[titleDF.index.isin(moviesML.index)]
    dict = {}
    for line in billing.itertuples():
        print(line)
        billstr = line.billing
        print(billstr)
        if billstr[0] == '"':
            billstr = billstr[1:-1]
        list = json.loads(billstr)
        actorList = []
        for item in list:
            actorList.append(item[2])
        dict[line.Index] = actorList
    return dict


def returnactorcoincidences(dictionary):
    """:type dictionary: dict"""
    coincidences = []
    for movie, actorlist in dictionary.items():
        for actor in actorlist:
            coincidences.extend(ratings[ratings['id'].str.contains(actor)]['id'].tolist())

    return coincidences


def getFullid(type, number):
    number = str(int(number))
    fullstr = "" + type
    for x in range(7 - len(number)):
        fullstr = fullstr + "0"
    fullstr = fullstr + str(number)
    return fullstr


def getRating(id):
    try:
        mRatings = ratings.set_index("id")
        return mRatings.loc[id]["rating"]
    except KeyError:
        print("Defaulting to avg rating for id: ", id)
        avgRatings = pd.DataFrame.from_csv("../sheets/ratings.tsv", sep="\t")
        return avgRatings.loc[id]["averageRating"]

# billingdfOG = pd.DataFrame.from_csv("../sheets/Processed/MoviesBilling2Trunc.tsv", sep="\t")
# billingdf = billingdfOG.sample(n=100)
# # TODO delete these two debug inputs
# billingdf = billingdf.append(billingdfOG.loc[114709])
# billingdf = billingdf.append(billingdfOG.loc[133093])
# actorDict = movieslookup(billingdf)
# actorcoincidences = returnactorcoincidences(actorDict)
