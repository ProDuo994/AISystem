import numpy as np
import string

data = {}
chars = set(string.ascii_letters)

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
    words = prompt.lower().split()  # simple tokenization
    promptData = {}
    for word in words:
        promptData[word] = promptData.get(word, 0) + 1
    return promptData


def submitQuery(query):
    if query:
        print("Analysing Query")
        QData = convertPromptToData(query)
        print("Processed Data:", QData)
        saveLearntData(QData)
    else:
        print("It seems that you have not submitted a prompt.")

nn.changeWeight(0.45)
i = input("> ")
submitQuery(i)
print("Accumulated Data:", data)
