import csv

with open("data/dataset.csv", "r", encoding="utf-8") as file:
    read = csv.reader(file)
    data = list(read)

happy_count = 0
sad_count = 0
anger_count = 0
fear_count = 0
disgust_count = 0

print("Total data count: ", len(data) - 1)
for i in range(1, len(data)):
    if data[i][1] == "happy":
        happy_count += 1
    if data[i][1] == "sad":
        sad_count += 1
    if data[i][1] == "anger":
        anger_count += 1
    if data[i][1] == "fear":
        fear_count += 1
    if data[i][1] == "disgust":
        disgust_count += 1

print("happy count: ", happy_count)
print("sad count: ", sad_count)
print("anger count: ", anger_count)
print("fear count: ", fear_count)
print("disgust count: ", disgust_count)