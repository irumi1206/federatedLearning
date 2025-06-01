import json
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filenamelist", nargs = "+", type = str)
args = parser.parse_args()


dataforeachfile = []

for filename in args.filenamelist:
    with open(f"{filename}/args.json") as f:
        data = json.load(f)
        dataforeachfile.append((filename,data))

fig, axes = plt.subplots(1,2,figsize=(12,5))

round = 100

for name, data in dataforeachfile:
    axes[0].plot(data["centralservertimepast"][:round], data["centralserveraccuracy"][:round],label=name)
axes[0].set_title("Accuracy per Time")
axes[0].set_xlabel("Time(msec)")
axes[0].set_ylabel("Accuracy(%)")
axes[0].legend()

for name, data in dataforeachfile:
    # if name == "0530_191836" or name == "0530_191838":
    #     axes[1].plot([i/5 for i in data["centralserverround"][:round]], data["centralserveraccuracy"][:round],label=name,marker='s')
    # else:


    axes[1].plot(data["centralserverround"][:round], data["centralserveraccuracy"][:round],label=name)
axes[1].set_title("Accuracy per round")
axes[1].set_xlabel("Round")
axes[1].set_ylabel("Accuracy(%)")
axes[1].legend()

plt.suptitle("Experiment Accuracy comparison")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")

