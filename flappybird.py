#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random

class FlappyBird:
    def __init__(self):
        # OUR VARIABLES
        self.collisionOn = True
        self.offSetDistance = 150
        self.dist = 0
        self.screenWidth = 400
        self.screenHeight = 708
        self.birdx = 70
        self.scalingFactor = 250
        
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
        if lowerPipeRect.colliderect(self.bird) & self.collisionOn:
            self.dead = True
        if downRect.colliderect(self.bird) & self.collisionOn:
            self.dead = True
        if not 0 < self.bird[1] < self.screenHeight:  
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.dist = 0
            self.wallx = 400
            self.offset = random.randint(-self.offSetDistance, self.offSetDistance)
            self.gravity = 5

    def run(self):
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 40)
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                    self.jump = 17
                    self.gravity = 5
                    self.jumpSpeed = 10

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.lowerPipe,
                             (self.wallx, self.lowerPipeTop()))
            self.screen.blit(self.upperPipe,
                             (self.wallx, self.upperPipeTop()))
            self.screen.blit(font.render("pipes: " + str(self.counter),
                                         -1,
                                         (255, 255, 255)),
                             (25, 50))
            self.screen.blit(font.render("dist: " + str(self.dist),
                                         -1,
                                         (255, 255, 255)),
                             (200, 50))
            if self.dead:
                self.sprite = 2
            elif self.jump:
                self.sprite = 1
            self.screen.blit(self.birdSprites[self.sprite], (self.birdx, self.birdY))
            if not self.dead:
                self.sprite = 0
            self.updateWalls()
            self.birdUpdate()
            pygame.display.update()

if __name__ == "__main__":
    FlappyBird().run()
