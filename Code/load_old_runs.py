import pickle

with open("saved_runs/test_run_3", "rb") as f:
    data = pickle.load(f)
    
print(data)