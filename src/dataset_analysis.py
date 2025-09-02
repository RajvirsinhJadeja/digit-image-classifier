import pickle
import csv

with open("data/word_map.pkl", "rb") as file:
    map = pickle.load(file)
    

with open("data/training.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    data = list(reader)

words_found = 0
total_words = 0

for row in data:
    list = row[0].split()
    
    for word in list:
        total_words += 1
        
        if map.get(word) is not None:
            words_found += 1

print(words_found/total_words * 100)

