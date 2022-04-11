import tensorflow
from tensorflow import keras
from dataPrep import prepData, unique_words
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv

labels, data = prepData('subtask1-heterographic-test.txt', 'subtask1-heterographic-test.xml')
labels2, data2 = prepData('subtask1-homographic-test.txt', 'subtask1-homographic-test.xml')

labels = labels + labels2
data = data + data2

train_data = data[:3224]

questions = []
with open('JEOPARDY_CSV.csv', 'r', encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    i = 0
    for row in csvreader:
        if i >= 5000:
            break
        i += 1
        questions.append(row[5])

train_data = data[:3224] + questions[:4000]

train_data = np.array(train_data)

num_unique = unique_words(train_data)

#tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=num_unique)
tokenizer.fit_on_texts(train_data)

model = tensorflow.keras.models.load_model("C:/Users/Epicr/OneDrive/Desktop/coding/punProject/punModel/")

import discord

TOKEN = 'ODU3NzA0MDEzMzc3MzA2Njc0.YNTdHA.-x8q0Uu2fMepOGgV-o9IP0vOGAw'

client = discord.Client()

@client.event
async def on_ready():
   print("Client: " + client.user.name + " is ready")

@client.event
async def on_message(message):
    if message.author == client.user:
      return

    if len(message.content.split()) > 5:
        msg = ''
        for i in message.content:
            if ',' != i and '.' != i and "'" != i and '"' != i and "\\" != i:
                msg = msg + i.lower()

        text = [msg]
        sequence = tokenizer.texts_to_sequences(text)
        padded_sequence = pad_sequences(sequence, maxlen=20, padding="post", truncating="post")

        predict = model.predict(padded_sequence)
        if predict < .5:
            print()
            print(message.author.name + " said: " + message.content)
            print("That is a PUN!")

            await message.channel.send(message.author.name + " has been caught using puns!")
client.run(TOKEN)