import numpy as np
import string
import asyncio

data = {}
chars = set(string.ascii_letters)

class Weight:
    def __init__(self, layer, weight):  # Fixed constructor
        self.layer = layer
        self.weight = weight

class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights

    def changeWeight(self, newWeight):
        self.weights = newWeight

nn = NeuralNetwork(0.23)

def saveLearntData(newData):
    global data
    for k, v in newData.items():
        data[k] = data.get(k, 0) + v

def convertPromptToData(prompt):
    words = prompt.lower().split()
    promptData = {}
    for word in words:
        promptData[word] = promptData.get(word, 0) + 1
    return promptData

async def submitQuery(query):
    if query:
        print("Analysing Query")
        QData = convertPromptToData(query)
        print("Processed Data:", QData)
        saveLearntData(QData)
    else:
        print("It seems that you have not submitted a prompt.")

def createLearningPoint(layer, weight):
    return Weight(layer, weight)

nn.changeWeight(0.45)

async def main():
    i = input("> ")
    await submitQuery(i)
    print("Accumulated Data:", data)

asyncio.run(main())
