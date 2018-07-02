import json
titleList = []
with open('../sheets/titlebasics.tsv') as infile, open('../JSON/titlebasics.json', 'w') as outfile:
    keys = ["id", "titleType", "primaryTitle", "originalTitle",
            "isAdult", "startYear", "endYear", "runtimeMinutes", "genres"]
    first = True
    for line in infile:
        if not first:
            title = {}
            tsplit = line.split("\t")
            for i,x in enumerate(tsplit):
                title[keys[i]] = x
            titleList.append(title)
            print(title)
        else:
            first = False
    infile.close()
    jsonList = json.dumps(titleList)
    outfile.write(jsonList)
    outfile.close()
