import math
import random
import numpy as np
import pygame
import os
from pygame import mixer
import argparse
import DQN
import keras.utils
import seaborn
import matplotlib.pyplot as plt
import statistics


#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1 / 100
    params['learning_rate'] = 0.005
    params['first_layer_size'] = 100  # neurons in the first layer
    params['episodes'] = 750
    params['memory_size'] = 3200
    # 32 default batch size in keras
    params['batch_size'] = 32
    params['random'] = False
    # Settings
    params['weights_path'] = 'weights/weights.hdf5'
    params['load_weights'] = False
    params['train'] = True
    params['plot_score'] = True
    params['computer_player'] = True
    params['display'] = False
    params['plot'] = True
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
    params['display'] = False
    mean, stddev = run(params)
    return mean, stddev


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
    # FPS = 60
    FPS = 120
    fpsClock = pygame.time.Clock()
    gametime = 1
    # create the screen
    xMax = 1600
    yMax = 1000

    # Background
    if params['display']:
        screen = pygame.display.set_mode((xMax, yMax))
        background = pygame.image.load('background.jpg')

        # Sound
        # mixer.music.load("darude.wav")
        # mixer.music.play(-1)

    # Caption and Icon
    pygame.display.set_caption("Rainy Day ")
    icon = pygame.image.load('player.png')
    pygame.display.set_icon(icon)
    counter_games = 0
    games_move_list = []
    agent = DQN.DQNAgent(params)
    weights_filepath = params['weights_path']
    if params['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    score_plot = []
    counter_plot = []

    while counter_games < params['episodes']:

        at_edge = False
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
        # playerX = 800
        playerX = 600
        playerY = yMax - playerSizeY

        playerX_change = 0

        # Enemy
        images = ['chromeBall.png', 'edgeBall.png', 'firefoxBall.png']
        enemyImg = []
        player_move_list = []
        enemyX = []
        enemyY = []
        enemyXVel = []
        enemyYVel = []
        enemyX_change = []
        enemyY_change = []
        num_of_enemies = 15
        cur_enemies = 1
        last_move = 0
        # pygame.init()

        total_score = 0

        for i in range(num_of_enemies):
            enemyImg.append(pygame.image.load(images[random.randint(0, 2)]))
            # enemyX.append(-100)
            enemyX.append(random.randint(-300, 100))
            enemyY.append(random.randint(0, 150))
            enemyXVel.append(random.randint(5, 9))
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
            if params['display']:
                # RGB = Red, Green, Blue
                screen.fill((0, 0, 0))
                # Background Image

                screen.blit(background, (0, 0))

            if params['computer_player']:

                state_old = agent.get_state(playerX, playerY, enemyX, enemyXVel, enemyY, playerX_change, enemyYVel,
                                            cur_enemies)

                # perform random actions based on agent.epsilon, or choose the action
                if random.uniform(0, 1) < agent.epsilon or params['random']:
                    final_move = keras.utils.to_categorical(random.randint(0, 2), num_classes=5)

                else:
                    # predict action based on the old state
                    prediction = agent.model.predict(state_old.reshape((1, DQN.num_inputs)), use_multiprocessing=True)
                    final_move = keras.utils.to_categorical(np.argmax(prediction[0]), num_classes=5)
                last_move = playerX_change
                playerX_change = player.do_move(final_move, playerX, playerY)
                # player_move_list.append(playerX_change)

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
                at_edge = True
            elif playerX >= xMax - playerSizeX:
                playerX = xMax - playerSizeX
                at_edge = True
            else:
                at_edge = False

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
                if enemyX[i] > xMax or enemyX[i] < -310:
                    enemyY[i] = random.randint(0, 150)
                    enemyYVel[i] = 5
                    enemyXVel[i] = -enemyXVel[i]
                    score_value = score_value + 1
                if params['display']:
                    screen.blit(enemyImg[i], (enemyX[i], enemyY[i]))
                # enemy(enemyX[i], enemyY[i], i)

                # Collision
                collision = isCollision(enemyX[i] + 64, enemyY[i] + 64, playerX + 85, playerY + 120)
                if collision:
                    running = False

            # if gametime % 3 == 0:
            state_new = agent.get_state(playerX, playerY, enemyX, enemyXVel, enemyY, playerX_change, enemyYVel,
                                        cur_enemies)
            reward = agent.set_reward(playerX, running, enemyX, enemyY, playerX_change, at_edge, playerY, enemyXVel,
                                      enemyYVel, last_move, cur_enemies)

            if params['train'] and not params['random']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, reward, state_new, game.crash)

            # player(curImage, playerX, playerY)
            if params['display']:
                screen.blit(curImage, (playerX, playerY))

            score = font.render("Score : " + str(score_value), True, (0, 0, 0))
            game = font.render("Game Number: " + str(counter_games), True, (0, 0, 0))
            high_score = font.render("High Score: " + str(record), True, (0, 0, 0))
            if params['display']:
                screen.blit(score, (textX, textY))
                screen.blit(game, (textX, textY - 40))
                screen.blit(high_score, (textX, textY - 80))

                # show_score(textX, textY)
                pygame.display.update()
            fpsClock.tick(FPS)

        if params['train'] and not params['random']:
            agent.replay_new(agent.memory, params['batch_size'])

        print('Game ' + str(counter_games) + ' Score ' + str(score_value))
        if score_value == 69:
            print('nice')
        score_plot.append(score_value)
        # games_move_list.append(player_move_list)
        counter_plot.append(counter_games)

    mean = statistics.mean(score_plot)
    stddev = statistics.stdev(score_plot)

    if params['train']:
        agent.model.save_weights(params['weights_path'])
        mean, stddev = test(params)

    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])

    return mean, stddev
    # if params['train']:
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


def plot_seaborn(array_counter, array_score, train):
    seaborn.set(color_codes=True, font_scale=1.5)
    seaborn.set_style("white")
    plt.figure(figsize=(13, 8))
    if train == False:
        fit_reg = False
    ax = seaborn.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        # color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg=fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)] * len(array_counter)
    ax.plot(array_counter, y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.ylim(0, 150)
    plt.show()


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
    mean, stddev = run(params)
    # mean,stddev = test(params)
    print('mean: ' + str(mean))
    print('std dev: ' + str(stddev))
