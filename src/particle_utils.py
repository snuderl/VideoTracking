import cv2

drawParticles = False
def initializeParticleSlider(pf, name, particleCount = 10, sigmaSpeed=6, sigmaSize=6):

	def setParticleCount(x):
		if x > 5:
			pf.count = x

	def setSigmaSpeed(x):
		if x > 0:
			pf.SIGMA_velocity = x / 10.

	def setSigmaSize(x):
		if x > 0:
			pf.SIGMA_size = x / 100.

	cv2.createTrackbar("Num particles", name, 5, 2000, setParticleCount)
	cv2.setTrackbarPos("Num particles", name, particleCount)

	cv2.createTrackbar("Sigma speed / 10", name, 1, 200, setSigmaSpeed)
	cv2.setTrackbarPos("Sigma speed / 10", name, sigmaSpeed)

	cv2.createTrackbar("Sigma size / 100", name, 0, 100, setSigmaSize)
	cv2.setTrackbarPos("Sigma size / 100", name, sigmaSize)

def drawParticles(image, pf):
    for particle in pf.particles:
        cx = int(particle[0] + particle[2] / 2)
        cy = int(particle[1] + particle[3] / 2)
        cv2.circle(image, (cx, cy), 1, 200, 2)

def initializeMeanshiftSlider(algo, name):
	def kernelSizeChanged(x):
		if x >= 3 and x % 2 == 1:
			algo.kernelSize = x
			algo.kernelChanged = True


	def sigmaChanged(x):
		if x >= 1 and x % 2 == 1:
			algo.kernelSigma = x
			algo.kernelChanged = True


	def scaleChanged(x):
		if x >= 1:
			algo.scale = x
		if x == 0:
			algo.scale = 0.5

	cv2.createTrackbar("Kernel size", name, 3, 11, kernelSizeChanged)
	cv2.setTrackbarPos("Kernel size", name, 3)
	cv2.createTrackbar("Sigma", name, 1, 10, sigmaChanged)
	cv2.setTrackbarPos("Sigma", name, 1)
	cv2.createTrackbar("Scale", name, 1, 10, scaleChanged)
	cv2.setTrackbarPos("Scale", name, 1)