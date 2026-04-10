import pickle

with open("file.pkl", "rb") as f:
    data = pickle.load(f)

print(data)
