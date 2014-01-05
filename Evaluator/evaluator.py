import subprocess
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


name = "Vid_A", "Vid_A_ball_gt"

directory = "../data/"
programDir = "../Evaluator/"

folder = "~/Desktop/project/VideoTracking/"


names = {
	"mean": "Mean-shift",
	"meanPF": "Mean-shift with particle filter",
	"PF": "Adaptive particle filter"
}
trackers = ["mean", "meanPF", "PF"]


script = programDir + "Evaluator.class"


def getScore(video, algo):
	files = [f for f in listdir(directory) if isfile(join(directory,f)) if f.startswith(video) and f.endswith(".txt")]
	f = files[0]
	reference = join(directory, f)
	fn = directory + "Tracked/" + video + algo + ".txt"
	print fn, reference
	os.system("java Evaluator " + reference + " " + fn)
	with open("../results.txt") as results:
		lines = results.readlines()
		mean = float(lines[2].split(" ")[1])
		scores = list(map(float, [x.split()[1] for x in lines[2:]]))
		return {
			"mean": mean,
			"scores": scores,
			"name": names[algo]
		}

def plotScores(title, scores):
	plt.title(title)
	plt.xlabel('Frame')
	plt.ylabel('Pokritost tarce(%)')
	for score in scores:
		label = "{} ({:.2f})".format(score["name"], score["mean"])
		plt.plot(range(0, len(score["scores"]))[::4], score["scores"][::4], label=label)
	a = scores[0]
	plt.plot([0, len(a["scores"])], [0.33, 0.33], "--", label="Minimalna sprejemljiva vrednost", linewidth=3)
	plt.legend()
	plt.show()

if __name__ == "__main__":
	import numpy as np
	mean = np.array([0.0,0,0])
	videos = ["A","B","C","D","E","F","G","H","I"]
	videos = ["C"]
	for x in videos:
		vid = "Vid_" + x
		a = getScore(vid, "mean")
		b = getScore(vid, "meanPF")
		c = getScore(vid, "PF")
		mean[0] += a["mean"]
		mean[1] += b["mean"]
		mean[2] += c["mean"]


		plotScores(vid, [a,b,c])

	#plt.plot(range(0, len(b["scores"]))[::4], b["scores"][::4])
	#plt.plot(range(0, len(c["scores"]))[::4], c["scores"][::4])

	print mean / len(videos)