# Code From https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a
# using this code to as starting version of DQNAgent and will modified as we go

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections
import math
import game

num_inputs = 16


def move_enemies(cur_enemies, enemyX, enemyXVel, enemyY, enemyYVel, yMax=1000, xMax=1600, loops=1):
    new_enemyX = list(enemyX)
    new_enemyY = list(enemyY)
    new_enemyXVel = list(enemyXVel)
    new_enemyYVel = list(enemyYVel)

    for moves in range(loops):
        for i in range(cur_enemies):

            # Ball movement
            new_enemyX[i] = new_enemyX[i] + new_enemyXVel[i]
            new_enemyY[i] = new_enemyY[i] + new_enemyYVel[i]
            new_enemyYVel[i] = new_enemyYVel[i] + 0.25

            if new_enemyY[i] > yMax - 128:
                new_enemyYVel[i] = -new_enemyYVel[i] * 0.85
                new_enemyY[i] = yMax - 128
            if new_enemyX[i] > xMax or new_enemyX[i] < -310:
                new_enemyY[i] = random.randint(0, 150)
                new_enemyYVel[i] = 5
                new_enemyXVel[i] = -enemyXVel[i]

    return enemyX, enemyY


class DQNAgent:
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        # will use keras sequential class since we have only one input and one output
        model = Sequential()
        model.add(Dense(self.first_layer, activation='sigmoid', input_dim=num_inputs))
        model.add(Dense(5, activation='sigmoid'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def get_state(self, playerX, playerY, enemyX, enemyXVel, enemyY, playerX_change, enemyYVel, cur_enemies):
        dist = 350
        dist2 = 900
        y_pos = 600
        one_move_from_death = False
        one_move_from_death_left = False
        one_move_from_death_right = False
        two_moves_from_death = False
        two_moves_from_death_left = False
        two_moves_from_death_right = False
        three_moves_from_death = False
        three_moves_from_death_left = False
        three_moves_from_death_right = False

        # state = [math.sqrt(math.pow(enemyX[i] - playerX, 2) + (math.pow(enemyY[i] - playerY, 2))) < dist and enemyY[i]
        #          >= y_pos for i in range(0,len(enemyX))]

        # state = [math.sqrt(math.pow(enemyX[i] - playerX, 2) + (math.pow(enemyY[i] - playerY, 2))) < dist for i in
        #         range(0, len(enemyX))]

        # state = [game.isCollision(enemyX[i]+enemyXVel[i],enemyY[i]+enemyYVel[i],playerX,playerY) for i in range(0,len(enemyX))]
        # state.extend([game.isCollision(enemyX[i] + enemyXVel[i], enemyY[i] + enemyYVel[i], playerX+7, playerY) for i in
        #         range(0, len(enemyX))])
        # state.extend([game.isCollision(enemyX[i] + enemyXVel[i], enemyY[i] + enemyYVel[i], playerX-7, playerY) for i in
        #         range(0, len(enemyX))])
        # state.extend([game.isCollision(enemyX[i] + enemyXVel[i], enemyY[i] + enemyYVel[i], playerX+14, playerY) for i in
        #         range(0, len(enemyX))])
        # state.extend([game.isCollision(enemyX[i] + enemyXVel[i], enemyY[i] + enemyYVel[i], playerX-14, playerY) for i in
        #         range(0, len(enemyX))])

        temp_moveX, temp_moveY = move_enemies(cur_enemies, enemyX, enemyXVel, enemyYVel, enemyYVel, loops=4)
        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX, playerY):
                one_move_from_death = True
        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX + 7, playerY):
                one_move_from_death_right = True
        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX - 7, playerY):
                one_move_from_death_left = True

        temp_moveX, temp_moveY = move_enemies(cur_enemies, enemyX, enemyXVel, enemyYVel, enemyYVel, loops=5)
        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX, playerY):
                two_moves_from_death = True
        temp_moveX, temp_moveY = move_enemies(cur_enemies, enemyX, enemyXVel, enemyYVel, enemyYVel)
        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX + 7, playerY):
                two_moves_from_death_right = True
        temp_moveX, temp_moveY = move_enemies(cur_enemies, enemyX, enemyXVel, enemyYVel, enemyYVel)
        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX - 7, playerY):
                two_moves_from_death_left = True

        temp_moveX, temp_moveY = move_enemies(cur_enemies, enemyX, enemyXVel, enemyYVel, enemyYVel, loops=6)

        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX, playerY):
                three_moves_from_death = True

        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX + 14, playerY):
                three_moves_from_death_right = True

        for i in range(0, len(enemyX)):
            if game.isCollision(temp_moveX[i], temp_moveY[i], playerX - 14, playerY):
                three_moves_from_death_left = True
        state = []
        state.extend([one_move_from_death, one_move_from_death_left, one_move_from_death_right, two_moves_from_death,
                      two_moves_from_death_left, two_moves_from_death_right, three_moves_from_death,
                      three_moves_from_death_left
                         , three_moves_from_death_right])

        state.append(playerX == 0)
        state.append(playerX == 1600)
        state.append(playerX_change == 0)
        state.append(playerX_change == 7)
        state.append(playerX_change == -7)
        state.append(playerX_change == 14)
        state.append(playerX_change == -14)
        # state.extend(
        #    [((playerX > enemyX[i] and enemyXVel[i] > 0) or (playerX < enemyX[i] and enemyXVel[i] < 0)) for i in
        #     range(0, len(enemyX))])

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self, playerX, running, enemyX, enemyY, playerX_change, at_edge, playerY, enemyXVec, enemyYVec,
                   last_move, cur_enemies):
        self.reward = 0
        if at_edge:
            self.reward -= 10
        if not running:
            self.reward -= 15
            return self.reward

        temp_moveX, temp_moveY = move_enemies(cur_enemies, enemyX, enemyY, enemyXVec, enemyYVec, loops=5)
        temp_moveX2, temp_moveY2 = move_enemies(cur_enemies, enemyX, enemyY, enemyXVec, enemyYVec, loops=6)
        temp_moveX3, temp_moveY3 = move_enemies(cur_enemies, enemyX, enemyY, enemyXVec, enemyYVec, loops=7)
        for i in range(cur_enemies):
            if (game.isCollision(temp_moveX[i], temp_moveY[i], playerX - playerX_change, playerY) or game.isCollision(
                    temp_moveX2[i], temp_moveY2[i], playerX - playerX_change, playerY) or game.isCollision(temp_moveX3[i], temp_moveY3[i], playerX - playerX_change, playerY)) and not at_edge:
                self.reward += 5

        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(np.array([next_state]), use_multiprocessing=True)[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0, use_multiprocessing=True)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(
                self.model.predict(next_state.reshape((1, num_inputs)), use_multiprocessing=True)[0])
        target_f = self.model.predict(state.reshape((1, num_inputs)), use_multiprocessing=True)
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, num_inputs)), target_f, epochs=1, verbose=0, use_multiprocessing=True)
