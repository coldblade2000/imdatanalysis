import pandas as pd


def billinglookup(type):
    ratings = pd.DataFrame.from_csv("../sheets/userratings.tsv", sep="\t")
    billing = pd.DataFrame.from_csv("../sheets/Processed/MoviesBilling.tsv", sep="\t")
    if type == 'tt':
        titleRatings = ratings.loc[ratings['type'] == 'tt']
        titleBilling = billing.isin(titleRatings)
        dict = {}
        for line in titleBilling.iterrows():
            dict[line['tconst']]: line["billing"]
        return dict
