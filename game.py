import math
import random
import numpy as np
import pygame
import os
from pygame import mixer
import argparse
import DQN
import keras.utils
import seaborn as sn


#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1 / 75
    params['learning_rate'] = 0.0005
    params['first_layer_size'] = 50  # neurons in the first layer
    params['second_layer_size'] = 300  # neurons in the second layer
    params['third_layer_size'] = 50  # neurons in the third layer
    params['episodes'] = 30
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.hdf5'
    params['load_weights'] = False
    params['train'] = True
    params['plot_score'] = True
    params['computer_player'] = True
    return params


class Game:
    def __init__(self, game_width, game_height):
        self.game_width = game_width
        self.game_height = game_height
        self.crash = False
        self.player = Player(1000)
        self.score = 0


class Player:

    def __init__(self, yMax):
        self.playerSizeX = 177
        self.playerSizeY = 239
        self.playerImgR = pygame.image.load('player.png')
        self.playerImgL = pygame.image.load('playerL.png')
        self.curImage = self.playerImgR
        self.playerX = 800
        self.playerY = yMax - self.playerSizeY

    def do_move(self, move, x, y):
        # move_array = [self.x_change, self.y_change]

        next_move = 0
        if np.array_equal(move, [1, 0, 0, 0, 0]):
            next_move = 0
            # move_array = self.x_change, self.y_change
            # move right
        elif np.array_equal(move, [0, 1, 0, 0, 0]):
            next_move = 1
            # move_array = [0, self.x_change]
            # move left
        elif np.array_equal(move, [0, 0, 1, 0, 0]):
            next_move = -1
            # move_array = [-self.y_change, 0]
            # sprint right
        elif np.array_equal(move, [0, 0, 0, 1, 0]):
            next_move = 2
            # move_array = [0, -self.x_change]
            # spring left
        elif np.array_equal(move, [0, 0, 0, 0, 1]):
            next_move = -2
            # move_array = [self.y_change, 0]
        # self.x_change, self.y_change = move_array
        # self.x = x + self.x_change
        # self.y = y + self.
        return 7 * next_move

        """
        if self.x < 20 or self.x > game.game_width - 40 \
                or self.y < 20 \
                or self.y > game.game_height - 40 \
                or [self.x, self.y] in self.position:
            game.crash = True
        """
def test(params):
    params['load_weights'] = True
    params['train'] = False

def isCollision(enemyX, enemyY, playerX, playerY):
    distance = math.sqrt(math.pow(enemyX - playerX, 2) + (math.pow(enemyY - playerY, 2)))
    if distance < 125:
        return True
    else:
        return False


def run(params):
    # Initialize the pygame

    x = 100
    y = 50
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
    record = 0
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
    counter_games = 0
    agent = DQN.DQNAgent(params)
    weights_filepath = params['weights_path']
    if params['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    while counter_games < params['episodes']:

        # if not training do not allow for random actions
        if not params['train']:
            agent.epsilon = 0.00
        else:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = 1 - (counter_games * params['epsilon_decay_linear'])

        counter_games += 1
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

        # pygame.init()

        score_plot = []
        counter_plot = []
        total_score = 0

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

        # Game Loop
        running = True
        game_over = False
        sprint = False
        while running:
            if score_value > record:
                record = score_value
            game = Game(440, 440)
            player = game.player
            # RGB = Red, Green, Blue
            screen.fill((0, 0, 0))
            # Background Image
            screen.blit(background, (0, 0))

            if params['computer_player']:

                state_old = agent.get_state(playerX,playerY, enemyX,enemyY)

                # perform random actions based on agent.epsilon, or choose the action
                if random.uniform(0, 1) < agent.epsilon:
                    final_move = keras.utils.to_categorical(random.randint(0, 2), num_classes=5)

                else:
                    # predict action based on the old state
                    prediction = agent.model.predict(state_old.reshape((1, 15)))
                    final_move = keras.utils.to_categorical(np.argmax(prediction[0]), num_classes=5)

                if gametime % 5 == 0:
                    playerX_change = player.do_move(final_move, playerX, playerY)
                    state_new = agent.get_state(playerX,playerY, enemyX,enemyY)
                    reward = agent.set_reward(score, running)

                    if params['train']:
                        # train short memory base on the new action and state
                        agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                        # store the new data into a long term memory
                        agent.remember(state_old, final_move, reward, state_new, game.crash)
            else:
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

                # Ball movement
                enemyX[i] = enemyX[i] + enemyXVel[i]
                enemyY[i] = enemyY[i] + enemyYVel[i]
                enemyYVel[i] = enemyYVel[i] + 0.25

                if enemyY[i] > yMax - 128:
                    enemyYVel[i] = -enemyYVel[i] * 0.85
                    enemyY[i] = yMax - 128
                if enemyX[i] > xMax or enemyX[i] < -128:
                    enemyY[i] = random.randint(0, 200)
                    enemyYVel[i] = 5
                    enemyXVel[i] = -enemyXVel[i]
                    score_value = score_value + 1
                screen.blit(enemyImg[i], (enemyX[i], enemyY[i]))
                # enemy(enemyX[i], enemyY[i], i)

                # Collision
                collision = isCollision(enemyX[i] + 64, enemyY[i] + 64, playerX + 85, playerY + 120)
                if collision:
                    game_over = True
                    running = False

            # player(curImage, playerX, playerY)
            screen.blit(curImage, (playerX, playerY))

            score = font.render("Score : " + str(score_value), True, (0, 0, 0))
            game = font.render("Game Number: " + str(counter_games), True, (0,0,0))
            high_score = font.render("High Score: " + str(record), True, (0, 0, 0))
            screen.blit(score, (textX, textY))
            screen.blit(game, (textX, textY-40))
            screen.blit(high_score, (textX, textY - 80))
            # show_score(textX, textY)
            pygame.display.update()
            fpsClock.tick(FPS)

        if params['train']:
            agent.replay_new(agent.memory, params['batch_size'])

    if params['train']:
        agent.model.save_weights(params['weights_path'])
        test(params)

    return record
        #if params['train']:
        #    agent.replay_new(agent.memory, params['batch_size'])

# dont think we need game over screen for loop
"""
    while (game_over):
        # RGB = Red, Green, Blue
        screen.fill((0, 0, 0))
        # Background Image
        screen.blit(background, (0, 0))
        score = font.render("Score : " + str(score_value), True, (0, 0, 0))
        screen.blit(score, (x, y))
        over_text = over_font.render("GAME OVER", True, (0, 0, 0))
        screen.blit(over_text, (200, 250))
        #show_score(textX, textY)
        #game_over_text()
        pygame.display.update()
        fpsClock.tick(FPS)
"""

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", type=bool, default=False)
    parser.add_argument("--speed", type=int, default=50)
    args = parser.parse_args()
    params['bayesian_optimization'] = False  # Use bayesOpt.py for Bayesian Optimization
    # run(args.display, args.speed, params)
    counter_games = 0
    record = run(params)
    print(record)
