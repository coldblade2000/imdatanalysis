import json

j = []

with open("./ExampleJSON", "r") as js:
    j = json.load(js)
#jstr = ""
#j = [x.strip("\n") for x in j]


#jsan = json.loads(j)

print(j["colors"])

