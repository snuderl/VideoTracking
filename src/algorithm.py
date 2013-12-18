import config
import learner
import particle
import haar_features as haar
import numpy as np
import utils
import pysomemodule
import signal

from utils import measureTime
from multiprocessing import Pool, cpu_count


abc = pysomemodule.ABC("ab")
def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

class Algorithm:
    def __init__(self, threads=1):
        self.adaBoost = learner.Trainer(config.ADA_BOOST_FEATURES)
        if threads>1:
            self.pool = Pool(threads, init_worker)
            self.threads = threads
        else:
            self.pool = None

        self.pos = None
        self.neg = None
        self.iterations = -1

    @property
    def particles(self):
        return self.pf.particles

    @property
    def target(self):
        return self.pf.target
    

    def start(self, image, target):
        # Initialize particles
        self.image = image.astype(np.float32)
        self.pf = particle.ParticleFilter(target, config.PARTICLES, image.shape[:2])
        # Generate haar features
        self.features = haar.generateHaarFeatures(config.FEATURES_COUNT).astype(np.float32)
        self.allFeatures = np.array(list(range(0, config.FEATURES_COUNT*3)))
        self.targetScore = 1
        self.generateNewSamples(config.NEG_INITIAL_SAMPLES)
        self.iterations = 0

    def next(self, image):
        self.image = image.astype(np.float32)
        pos, neg = self.pos, self.neg
        with measureTime("Ada boost learning"):
            train = np.vstack((pos, neg))
            targets = np.zeros((pos.shape[0] + neg.shape[0]))
            targets[:pos.shape[0]] = 1  
            weights = np.ones(targets.shape)
            weights[0] = 20
            weights = weights / weights.sum()
            self.adaBoost.train(train, targets, weights)
            self.indices = self.adaBoost.features()

        with measureTime("Updating particles"):
            self.pf.updateParticles()
        with measureTime("Calculating particle features"):
            particle_features = self.calculateParticleFeatures()
        with measureTime("Scoring particles:"):
            probabilities = self.adaBoost.predict(particle_features)
        with measureTime("Updating weights:"):
            self.pf.updateWeights(probabilities)
        with measureTime("Scoring target particle:"):
            targetVector = abc.doSomething(
                image,
                np.array([self.target]).astype(np.float32),
                self.features,
                self.indices)
            self.targetScore = self.adaBoost.predict(targetVector)
            print "Score {0}, ".format(self.targetScore)
        with measureTime("Generating new samples:"):
            self.generateNewSamples(config.NEW_SAMPLES_PER_ITERATION)


        self.iterations += 1

    def generateNewSamples(self, newSamples):

        if self.targetScore > config.TRESHOLD_PROB:
            positive = self.pf.target.reshape((1, 4))
            generator = utils.rectangleGenerator(
                self.image.shape[1],
                self.image.shape[0],
                newSamples,
                invalid=self.pf.target)
            negative = np.array(list(generator))
            examples = np.vstack((negative, positive))

            examples = examples.astype(np.float32)

            feature_vector = abc.doSomething(
                self.image, examples, self.features, self.allFeatures)
            #print feature_vector, feature_vector.dtype
            if not self.neg == None:
                if self.neg.shape[0] < config.NEG_EXAMPLES:
                    self.neg = np.vstack((self.neg, feature_vector[:-1, :]))
                else:
                    self.neg[np.random.randint(0, self.neg.shape[0], newSamples)] = feature_vector[:-1, :]
                positive = feature_vector[-1, :].reshape(1, feature_vector.shape[1])

                if self.pos.shape[0] >= config.POS_EXAMPLES:
                    self.pos[np.random.randint(1, config.POS_EXAMPLES), :] = positive
                else:
                    self.pos = np.vstack((self.pos, positive))

            else:
                self.neg = feature_vector[:-1, :]
                positive = feature_vector[-1, :].reshape(1, feature_vector.shape[1])
                self.pos = positive
        else:
            print "Probability of estimation is low. Not updating train data."



    def calculateParticleFeatures(self):
        image = self.image
        particles = self.particles.astype(np.float32)
        if self.pool:
            results = []
            size = self.threads/self.threads
            for x in range(self.threads):
                if x==self.threads-1:
                    results.append(self.pool.apply_async(call, [image, particles[size*x:,:], self.features, self.indices]))
                else:
                    results.append(self.pool.apply_async(call, [image, particles[size*x:size*(x+1),:], self.features, self.indices]))
            particle_features = np.vstack((map(lambda x: x.get(), results)))
        else:
            particle_features = pysomemodule.ABC("ab").doSomething(
                image, particles, self.features, self.indices)
        return particle_features

def call(*args):
    features = pysomemodule.ABC("ab").doSomething(*args)
    return features




