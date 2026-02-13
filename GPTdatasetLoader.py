from datasets import load_dataset, concatenate_datasets
import numpy as np
import random

def loadDataset():
    tinyTrain = load_dataset("starhopp3r/TinyChat", split="train[500:]")
    tinyTest  = load_dataset("starhopp3r/TinyChat", split="train[:500]")
    chatTrain = load_dataset("raincandy-u/TinyChat", split="train[500:]")
    chatTest = load_dataset("raincandy-u/TinyChat", split="train[:500]")
    englishTrain = load_dataset("agentlans/high-quality-english-sentences", split="train")
    englishTest = load_dataset("agentlans/high-quality-english-sentences", split="test[:500]")
    train_dataset = concatenate_datasets([tinyTrain, chatTrain, englishTrain])
    test_dataset = concatenate_datasets([tinyTest, chatTest, englishTest])
    return train_dataset, test_dataset

def forceToMaxLength(array, maxLength):
    if len(array) > maxLength:
        offset = random.randint(0, len(array) - maxLength - 1)
        array = array[offset:offset + maxLength + 1]
    else:
        array.reverse()
        while len(array) <= maxLength:
            array.append(0)
        array.reverse()
    return array
def getDataEntry(array, i, tokenizer, max_length):
    text = array[i]['text'] + "[END]"
    text = text.replace("<A>", "[INST]").replace("<B>", "[/INST]").replace("<end>", "")
    entry = tokenizer.tokenize(text)
    if (entry[0] != 0):
        entry = [0] + entry
    entry = forceToMaxLength(entry, max_length + 1)
    x = entry[:-1]
    y = entry[1:]
    return (x,y)
def dataloader(array, batch_size, tokenizer, max_length):
    dataset_size = array.shape[0]
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            batch = []
            batch_y = []
            for i in range(len(batch_perm)):
                x, y = getDataEntry(array, batch_perm[i], tokenizer, max_length)
                batch.append(x)
                batch_y.append(y)
            yield (np.array(batch), np.array(batch_y))
            start = end
            end = start + batch_size

def getDataLoaders(BATCH_SIZE, tokenizer, max_length):
    train_dataset, test_dataset = loadDataset()
    train_dataloader = dataloader(train_dataset, BATCH_SIZE, tokenizer, max_length)
    test_dataloader = dataloader(test_dataset, BATCH_SIZE, tokenizer, max_length)
    return train_dataloader, test_dataloader