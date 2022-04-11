import xml.etree.ElementTree as ET
import numpy as np
def prepData(ex, jokes):
    #Prepare expected
    labels = []
    with open(ex) as f:
        for line in f.readlines():
            labels.append(float(line.split()[1]))

    #Prepare jokes
    tree = ET.parse(jokes)
    root = tree.getroot()

    jokes = []

    s = ''
    for joke in root:
        for word in joke:
            if ',' not in word.text and '.' not in word.text and "'" not in word.text and '"' not in s and "\\" not in s and '$' not in s:
                s = s + word.text.lower() + ' '
        jokes.append(s)
        s = ''

    return labels, jokes

def unique_words(data):
    words = []
    for joke in data:
        for word in joke.split():
            if word not in words:
                words.append(word)
    return len(words)
