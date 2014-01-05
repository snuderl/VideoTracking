import subprocess
import os


name = "Vid_A", "Vid_A_ball_gt"

directory = "../data/"
programDir = "../Evaluator/"

folder = "~/Desktop/project/VideoTracking/"

trackers = ["mean", "meanPF", "PF"]

script = programDir + "Evaluator.class"
reference = directory + name[1] + ".txt"
with open(reference) as ref:
	for x in trackers:
		fn = directory + "Tracked/" + name[0] + x + ".txt"

		print script
		os.system("cd " + programDir + "; cd Evaluator")
