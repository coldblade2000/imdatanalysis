import json
import timeit
import pandas as pd

id = "tt8630480"

def iterate():
    with open("../JSON/titlebasics.json") as infile:
        lis = json.load(infile)
        for object in lis:
            print(object["id"])
            if object["id"] == id:
                print("Done")
                break
        infile.close()

def turnIntoDic():
    with open("../JSON/titlebasics.json") as infile, open("../JSON/titlebasicsDic.json","w") as outfile:
        lis = json.load(infile)
        dic = {}
        for object in lis:
            dic[object["id"]]=object
        outfile.write(json.dumps(dic))
        outfile.flush()
        outfile.close()

def find():
    with open("../JSON/titlebasicsDic.json") as infile:
        dic = json.load(infile)
        print(dic[id])

def processSheets():
    titlebasics = pd.DataFrame.from_csv("../sheets/titlebasics.tsv", sep="\t", header=0)
    ratings = pd.DataFrame.from_csv("../sheets/ratings.tsv", sep="\t", header=0)
    crew = pd.DataFrame.from_csv("../sheets/crew.tsv", sep="\t", header=0)
    episode = pd.DataFrame.from_csv("../sheets/episode.tsv", sep="\t", header=0)
    # namebasics = pd.DataFrame.from_csv("../sheets/namebasics.tsv", sep="\t", header=0)
    # principals = pd.DataFrame.from_csv("../sheets/principals.tsv", sep="\t", header=0)
    # ui.toUserInterface([{
    #     "id":"tt89490"
    #     "score":
    #                      }])
    # frame1.info()
    # frame2.info()
    result = titlebasics.join(ratings)
    result = result.join(crew)
    result = result.join(episode)
    result.drop("titleType", axis=1, inplace=True)
    print(result.tail(3))
    pd.DataFrame.to_csv(result, "../sheets/Processed/Titles.tsv", sep="\t")

titleList=[]
with open("../sheets/principals.tsv") as openfile:
    first = True
    keys=["tconst", "ordering", "nconst", "category", "job", "characters"]
    for line in openfile:
        if (not first):
            title = {}
            tsplit = line.split("\t")
            for i, x in enumerate(tsplit):
                title[keys[i]] = x


        else:
            first = False
    openfile.close()
#titleList.reverse()
currentid = ""
individualTitleArray = []
titleDF = pd.DataFrame.from_csv("../sheets/Processed/Titles.tsv", sep="\t")
#titleDF['billing'] = []
newColumn = pd.DataFrame(index=titleDF.index)
titleDF.join(newColumn)
for i, item in enumerate(titleList):
    if item["tconst"] == currentid:
        individualTitleArray.append(item)
    elif currentid == "":
        currentid = item["tconst"]
        individualTitleArray.append(item)
    else:
        titleDF.at[currentid,"billing"] = individualTitleArray
        print(individualTitleArray)
        individualTitleArray=[item]
        currentid = item["tconst"]

titleDF.to_csv("../sheets/Processed/TitlesExp.tsv", sep="\t")