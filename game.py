import math
import random
import numpy as np
import pygame
import os
from pygame import mixer

class player:
    def do_move(self, move, x, y, game, food, agent):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1
            #do nothing
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
            # move right
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
            #move left
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
            #sprint right
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
            # spring left
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x = x + self.x_change
        self.y = y + self.y_change

        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or [self.x, self.y] in self.position:
            game.crash = True




# Initialize the pygame
x = 100
y = 50
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)

pygame.init()
FPS = 60
fpsClock = pygame.time.Clock()
gametime = 1
# create the screen
xMax = 1600
yMax = 1000
screen = pygame.display.set_mode((xMax, yMax))

# Background
background = pygame.image.load('background.jpg')


# Sound
mixer.music.load("darude.wav")
mixer.music.play(-1)

# Caption and Icon
pygame.display.set_caption("Rainy Day ")
icon = pygame.image.load('player.png')
pygame.display.set_icon(icon)

# Player
playerSizeX = 177
playerSizeY = 239
playerImgR = pygame.image.load('player.png')
playerImgL = pygame.image.load('playerL.png')
curImage = playerImgR
playerX = 800
playerY = yMax - playerSizeY

playerX_change = 0

# Enemy
images = ['chromeBall.png', 'edgeBall.png', 'firefoxBall.png']
enemyImg = []
enemyX = []
enemyY = []
enemyXVel = []
enemyYVel = []
enemyX_change = []
enemyY_change = []
num_of_enemies = 15
cur_enemies = 1

for i in range(num_of_enemies):
    enemyImg.append(pygame.image.load(images[random.randint(0, 2)]))
    enemyX.append(-100)
    enemyY.append(random.randint(0, 200))
    enemyXVel.append(random.randint(4, 10))
    enemyYVel.append(5)


# Score

score_value = 0
font = pygame.font.Font('freesansbold.ttf', 32)

textX = 100
textY = 100

# Game Over
over_font = pygame.font.Font('freesansbold.ttf', 64)


def show_score(x, y):
    score = font.render("Score : " + str(score_value), True, (0, 0, 0))
    screen.blit(score, (x, y))

def game_over_text():
    over_text = over_font.render("GAME OVER", True, (0, 0, 0))
    screen.blit(over_text, (200, 250))

def player(playerImg, x, y):
    screen.blit(playerImg, (x, y))

def enemy(x, y, i):
    screen.blit(enemyImg[i], (x, y))

def isCollision(enemyX, enemyY, playerX, playerY):
    distance = math.sqrt(math.pow(enemyX - playerX, 2) + (math.pow(enemyY - playerY, 2)))
    if distance < 125:
        return True
    else:
        return False

# Game Loop
running = True
game_over = False
sprint = False
while running:

    # RGB = Red, Green, Blue
    screen.fill((0, 0, 0))
    # Background Image
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # if keystroke is pressed check whether its right or left
        keys = pygame.key.get_pressed()
        playerX_change = 0
        if keys[pygame.K_LSHIFT]:
            sprint = True
        else:
            sprint = False
        if keys[pygame.K_LEFT]:
            playerX_change = -7 - int(sprint) * 7
            curImage = playerImgL
        if keys[pygame.K_RIGHT]:
            playerX_change = 7 + int(sprint) * 7
            curImage = playerImgR

        """if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                playerX_change = -7 - int(sprint) * 7
                curImage = playerImgL
            if event.key == pygame.K_RIGHT:
                playerX_change =  7 + int(sprint) * 7
                curImage = playerImgR
            if event.key == pygame.K_LSHIFT:
                sprint = True

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                playerX_change = 0
            if event.key == pygame.K_LSHIFT:
                sprint = False """

    playerX += playerX_change
    if playerX <= 0:
        playerX = 0
    elif playerX >= xMax - playerSizeX:
        playerX = xMax - playerSizeX

    # Enemy Movement
    gametime = gametime + 1
    if gametime % 460 == 0 and cur_enemies < num_of_enemies:
        cur_enemies = cur_enemies + 1
    for i in range(cur_enemies):

        #Ball movement
        enemyX[i] = enemyX[i] + enemyXVel[i]
        enemyY[i] = enemyY[i] + enemyYVel[i]
        enemyYVel[i] = enemyYVel[i] + 0.25

        if enemyY[i] > yMax-128:
            enemyYVel[i] = -enemyYVel[i] * 0.85
            enemyY[i] = yMax-128
        if enemyX[i] > xMax or enemyX[i] < -128:
            enemyY[i] = random.randint(0, 200)
            enemyYVel[i] = 5
            enemyXVel[i] = -enemyXVel[i]
            score_value = score_value + 1
        enemy(enemyX[i], enemyY[i], i)

        # Collision
        collision = isCollision(enemyX[i]+64, enemyY[i]+64, playerX+85, playerY+120)
        if collision:
            game_over = True
            running = False

    player(curImage, playerX, playerY)
    show_score(textX, textY)
    pygame.display.update()
    fpsClock.tick(FPS)

while(game_over):
    # RGB = Red, Green, Blue
    screen.fill((0, 0, 0))
    # Background Image
    screen.blit(background, (0, 0))
    show_score(textX, textY)
    game_over_text()
    pygame.display.update()
    fpsClock.tick(FPS)