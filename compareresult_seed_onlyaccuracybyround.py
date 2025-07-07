import json
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--filenamelist", nargs = "+", type = str)
parser.add_argument("--randomseeds", nargs = "+", type = int)
args = parser.parse_args()


dataforeachfile = []
recordstamp = 100

for filename in args.filenamelist:
    averagefiledata = defaultdict(list)
    averagefiledata["round"] = [0.0] * recordstamp
    averagefiledata["accuracy"] = [0.0] * recordstamp

    for randomseed in args.randomseeds:
        with open(f"{filename}#{randomseed}/args.json") as f:
            data = json.load(f)
            for i in range(min(recordstamp,len(data["centralserverround"]))):
                averagefiledata["round"][i] += data["centralserverround"][i]
                averagefiledata["accuracy"][i] += data["centralserveraccuracy"][i]

    averagefiledata["round"] = [x / len(args.randomseeds) for x in averagefiledata["round"]]
    averagefiledata["accuracy"] = [x / len(args.randomseeds) for x in averagefiledata["accuracy"]]

    dataforeachfile.append((filename,averagefiledata))

fig, axes = plt.subplots(1,1,figsize=(12,5))


for name, data in dataforeachfile:
    print(data)
    axes.plot(data["round"], data["accuracy"],label=name)
axes.set_title("Accuracy per Round")
axes.set_xlabel("Round")
axes.set_ylabel("Accuracy(%)")
axes.legend()

# for name, data in dataforeachfile:
#     axes[1].plot(data["centralserverround"][:round], data["centralserveraccuracy"][:round],label=name)
# axes[1].set_title("Accuracy per round")
# axes[1].set_xlabel("Round")
# axes[1].set_ylabel("Accuracy(%)")
# axes[1].legend()


plt.suptitle("Experiment Accuracy comparison")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")

