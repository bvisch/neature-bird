#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random
from collections import deque

import tensorflow as tf
import numpy as np

class FlappyBird:
    def __init__(self):
        # OUR VARIABLES
        self.collisionOn = False
        self.offSetDistance = 150
        self.dist = 0
        self.screenWidth = 400
        self.screenHeight = 720
        self.birdx = 70
        self.scalingFactor = 250
        self.FALL = 0
        self.JUMP = 1

        # EXISTING VARIABLES
        self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight))
        self.bird = pygame.Rect(50, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.lowerPipe = pygame.image.load("assets/bottom.png").convert_alpha()
        self.upperPipe = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 160
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-self.offSetDistance, self.offSetDistance)

        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 40)

    def upperPipeTop(self):
        return 0 - self.gap - self.offset

    def upperPipeBottom(self): # aka gapTop
        return 0 - self.gap - self.offset + self.upperPipe.get_height()

    def lowerPipeTop(self): # aka gapBottom
        return 360 + self.gap - self.offset

    def pipeLeftSide(self): # aka gapLeft
        return self.wallx + 2

    def pipeRightSide(self): # aka gapRight
        return self.pipeLeftSide() + self.upperPipe.get_width()

    def distFromGapTop(self):
        return self.upperPipeBottom() - self.birdY

    def distFromGapBottom(self):
        return self.lowerPipeTop() - self.birdY

    def distFromGapLeft(self):
        return self.pipeLeftSide() - self.birdx

    def distFromGapRight(self):
        return self.pipeRightSide() - self.birdx

    def distFromScreenBottom(self):
        return self.screenHeight - self.birdY

    def inGoodState(self):
        return self.upperPipeBottom() < self.birdY < self.lowerPipeTop()

    def reward(self): # ~200 dist traveled for each pipe
        return self.scalingFactor * self.counter + self.dist

    def updateWalls(self):
        self.wallx -= 2
        self.dist += 1
        if self.wallx < -self.upperPipe.get_width():
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-self.offSetDistance, self.offSetDistance)

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        lowerPipeRect = pygame.Rect(self.wallx,
                             self.lowerPipeTop() + 10,
                             self.lowerPipe.get_width() - 10,
                             self.lowerPipe.get_height())
        downRect = pygame.Rect(self.wallx,
                               self.upperPipeTop() - 10,
                               self.upperPipe.get_width() - 10,
                               self.upperPipe.get_height())
        if (lowerPipeRect.colliderect(self.bird) & self.collisionOn
            or downRect.colliderect(self.bird) & self.collisionOn
            or not 0 < self.bird[1] < self.screenHeight):
            self.dead = True

    def reset(self):
        self.bird[1] = 350
        self.birdY = 350
        self.dead = False
        self.counter = 0
        self.dist = 0
        self.wallx = 400
        self.offset = random.randint(-self.offSetDistance, self.offSetDistance)
        self.gravity = 5

    def drawEnvironment(self):
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.lowerPipe,
                         (self.wallx, self.lowerPipeTop()))
        self.screen.blit(self.upperPipe,
                         (self.wallx, self.upperPipeTop()))
        self.screen.blit(self.font.render("pipes: " + str(self.counter),
                                     -1,
                                     (255, 255, 255)),
                         (25, 50))
        self.screen.blit(self.font.render("dist: " + str(self.dist),
                                     -1,
                                     (255, 255, 255)),
                         (200, 50))

    def step(self, n, action, maxDist):
        dead = False
        for i in xrange(n):
            self.clock.tick(60)

            dead |= self.dead
            if not self.dead:
                self.sprite = 0
                if action == self.JUMP:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpSpeed = 10
            else:
                self.sprite = 2

            if self.jump:
                self.sprite = 1

            self.drawEnvironment()
            self.screen.blit(self.birdSprites[self.sprite], (self.birdx, self.birdY))
            self.updateWalls()
            self.birdUpdate()
            dead |= self.dead
            pygame.display.update()

        # reward = self.dist + self.counter*10
        reward = 1 if not dead else -100
        return [self.getState(), reward, self.dist]

    def getState(self):
        return [self.birdY/self.screenHeight, self.dead, self.jump/17, self.dist/50]
                # self.upperPipeBottom(),
                # self.lowerPipeTop(),
                # self.pipeLeftSide(),
                # self.pipeRightSide()]


game = FlappyBird()

nInputs = len(game.getState())
nOutputs = 2
layer1 = 4
layer2 = 2
REPLAY_SIZE = 1600
BATCH_SIZE = 800

inputs = tf.placeholder(shape=[None,nInputs], dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([nInputs,layer1], 0, 0.01))
b1 = tf.Variable(tf.zeros([layer1]))
hidden = tf.nn.sigmoid(tf.matmul(inputs, W1) + b1)
# W2 = tf.Variable(tf.random_uniform([layer1,layer2], 0, 0.01))
# b2 = tf.Variable(tf.zeros([layer2]))
# hidden2 = tf.nn.sigmoid(tf.matmul(hidden, W2) + b2)
# W3 = tf.Variable(tf.random_uniform([layer2,nOutputs], 0, 0.01))
# b3 = tf.Variable(tf.zeros([nOutputs]))
# Qout = tf.reshape(tf.nn.softmax(tf.matmul(hidden2, W3)), [nOutputs])
W2 = tf.Variable(tf.random_uniform([layer1,nOutputs], 0, 0.01))
b2 = tf.Variable(tf.zeros([nOutputs]))
Qout = tf.nn.softmax(tf.matmul(hidden, W2))
predict = tf.reshape(tf.argmax(Qout, 1), [])
maxQVal = tf.reduce_max(Qout, reduction_indices=1)

# nextQ = tf.placeholder(shape=[1,nOutputs], dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(nextQ - maxQVal))
# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# updateModel = trainer.minimize(loss)

actionTaken = tf.placeholder(shape=(None,nOutputs), dtype = tf.float32)
y = tf.placeholder(shape=(None,1), dtype=tf.float32)
actionValue = tf.reduce_sum(tf.mul(Qout, actionTaken), reduction_indices=1)
cost = tf.reduce_mean(tf.square(y - actionValue))
trainer = tf.train.AdamOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(cost)


if __name__ == "__main__":

    GAMMA = .99
    EPSILON = 0.1
    EPOCHS = 1000

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    maxr = 0
    maxDist = 0
    replayMemory = deque(maxlen=REPLAY_SIZE)

    for i in xrange(EPOCHS):
        game.reset()
        state = game.getState()

        while not game.dead:
            action, Q = sess.run([predict, Qout], feed_dict={ inputs: np.array([state], dtype=np.float32) })
            print str(action) + " " + str(Q)
            if np.random.rand(1) < EPSILON:
                action = np.random.randint(2)

            newState, reward, dist = game.step(2, action, maxDist)

            actionTensor = [0,0]
            actionTensor[action] = 1
            replayMemory.append((state, actionTensor, reward, newState))
            if len(replayMemory) >= REPLAY_SIZE:
                minibatch = np.array(random.sample(replayMemory, BATCH_SIZE))
                states = np.vstack(minibatch[:,0])
                actions = np.vstack(minibatch[:,1])
                rewards = minibatch[:,2]
                newStates = np.vstack(minibatch[:,3])

                maxQBatch = sess.run(maxQVal, feed_dict={ inputs: newStates })
                yBatch = []
                for i in xrange(BATCH_SIZE):
                    if rewards[i] < 0:
                        yBatch.append([rewards[i]])
                    else:
                        yBatch.append([rewards[i] + GAMMA * maxQBatch[i]])

                sess.run(updateModel, feed_dict={ inputs: states, actionTaken: actions, y: yBatch })


            # maxQ = sess.run(maxQVal, feed_dict={ inputs: np.array([newState], dtype=np.float32) })
            # targetQ = Q
            # targetQ[action] = reward + GAMMA*maxQ
            # sess.run(updateModel, feed_dict={ inputs: np.array([state], dtype=np.float32), nextQ: targetQ.reshape(1, nOutputs) })
            # print sess.run(W2, feed_dict={ inputs: np.array([state], dtype=np.float32) })
            state = newState
            # if dist > maxDist:
            #     maxDist = dist
            #     print dist
            # if reward > maxr:
            #     maxr = reward
            #     print maxr
