import pickle

with open('RunModel.pkl', 'rb') as imp:
    RunModel = pickle.load(imp)
    RunModel("/models/GPT_1.4.eqx")