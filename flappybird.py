#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random

import tensorflow as tf
import numpy as np

class FlappyBird:
    def __init__(self):
        # OUR VARIABLES
        self.collisionOn = False;
        self.offSetDistance = 150;
        self.dist = 0;

        # EXISTING VARIABLES
        self.screen = pygame.display.set_mode((400, 708))
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

    def upperPipeBottom(self): # also gapTop
        return 0 - self.gap - self.offset + self.upperPipe.get_height()

    def lowerPipeTop(self): # also gapBottom
        return 360 + self.gap - self.offset

    def pipeLeftSide(self): # also gapLeft
        return self.wallx + 2

    def pipeRightSide(self): # also gapRight
        return self.pipeLeftSide() + self.upperPipe.get_width()

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
        if lowerPipeRect.colliderect(self.bird) & self.collisionOn:
            self.dead = True
        if downRect.colliderect(self.bird) & self.collisionOn:
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.dist = 0
            self.wallx = 400
            self.offset = random.randint(-self.offSetDistance, self.offSetDistance)
            self.gravity = 5

    def step(self, n, action):
        for i in range(n):
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                # if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                #     self.jump = 17
                #     self.gravity = 5
                #     self.jumpSpeed = 10

            if action and not self.dead:
                self.jump = 17
                self.gravity = 5
                self.jumpSpeed = 10

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
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            pygame.display.update()

        reward = self.dist + self.counter*10
        return [self.getState(), reward]

    def getState(self):
        return [self.birdY,
                self.upperPipeBottom(),
                self.lowerPipeTop(),
                self.pipeLeftSide(),
                self.pipeRightSide()]


inputs = tf.placeholder(shape=[1,5], dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([5,50], 0, 0.01))
b1 = tf.Variable(tf.zeros([50]))
hidden = tf.nn.relu(tf.matmul(inputs, W1) + b1)
W2 = tf.Variable(tf.random_uniform([50,1], 0, 0.01))
b2 = tf.Variable(tf.zeros([1]))
Qout = tf.nn.relu(tf.matmul(hidden, W2) + b2)


nextQ = tf.placeholder(shape=[1,1], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

if __name__ == "__main__":
    game = FlappyBird()

    y = .99
    e = 0.1
    epochs = 1000

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    maxr = 0

    for i in xrange(epochs):
        s = game.getState()

        while True:
            a = sess.run(Qout, feed_dict={ inputs: np.array([s], dtype=np.float32) })
            if np.random.rand(1) < e:
                a = np.random.randint(2)

            s1, r = game.step(2, a)
            Q1 = sess.run(Qout, feed_dict={ inputs: np.array([s1], dtype=np.float32) })
            target = r + y*Q1[0]
            sess.run(updateModel, feed_dict={ inputs: np.array([s], dtype=np.float32), nextQ: np.array([target], dtype=np.float32) })

            s = s1
            if r > maxr:
                maxr = r
                print maxr
