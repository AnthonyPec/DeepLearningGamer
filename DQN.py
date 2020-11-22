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

num_inputs = 34


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
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Dense(self.first_layer, activation='relu', input_dim=num_inputs))
        model.add(Dense(self.second_layer, activation='relu'))
        model.add(Dense(self.third_layer, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model

    def get_state(self, playerX,playerY, enemyX,enemyXVel,enemyY,playerX_change):
        dist = 500
        y_pos = 600

       # state = [math.sqrt(math.pow(enemyX[i] - playerX, 2) + (math.pow(enemyY[i] - playerY, 2))) < dist and enemyY[i]
       #          >= y_pos for i in range(0,len(enemyX))]

        state = [math.sqrt(math.pow(enemyX[i] - playerX, 2) + (math.pow(enemyY[i] - playerY, 2))) < dist for i in range(0,len(enemyX))]
        state.append(playerX_change == 1)
        state.append(playerX_change == -1)
        state.append(playerX_change == 2)
        state.append(playerX_change == -2)
        state.extend([((playerX > enemyX[i] and enemyXVel[i] > 0) or (playerX < enemyX[i] and enemyXVel[i] < 0)) and abs(playerX - enemyX[i]) < 500 for i in range(0, len(enemyX))])

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self,playerX,running,enemyX,enemyY):
        self.reward = 0
        if not running:
            self.reward = -10
            return self.reward
        for i in range(0, len(enemyX)):
            if (abs(playerX + 89 - enemyX[i] + 64) < 200) and (enemyY[i] < 650):
                self.reward += 1
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
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=10, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, num_inputs)))[0])
        target_f = self.model.predict(state.reshape((1, num_inputs)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, num_inputs)), target_f, epochs=2, verbose=0)