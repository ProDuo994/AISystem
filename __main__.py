import numpy as np
import string

data = {}
chars = set(string.ascii_letters)

class NueralNetwork:
  def changeWeight(newWeight):
    self.weights = newWeight
  def __init__(self, weights):
    self.weights = weights

nn = NeuralNetwork(0.23)

def saveLearntData(data):
  d = set(data)

def convertPromptToData(prompt):
    promptData = {} 
    
    for l in prompt:
        if l in chars: 
            promptData[l] = promptData.get(l, 0) + 1
    return promptData 

def submitQuery(query):
    if query:
        print("Analysing Query")
        QData = convertPromptToData(query)
        print("Processed Data:", QData)
    else:
        print("It seems that you have not submitted a prompt.")
    saveLearntData(data)

i = input("> ")
submitQuery(i)
