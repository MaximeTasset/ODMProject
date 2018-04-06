# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:12:23 2018

@author: Sarah
"""
from pacman import Directions
from game import Agent
from ghostAgents import GhostAgent
import util
from sklearn.neural_network import MLPRegressor
import numpy as np



class ReinfAgent(GhostAgent,Agent):

    def __init__(self, index=0,epsilon=0.1):

        self.lastMove = Directions.STOP
        self.index = index

        self.learning_algo = None
        self.learn = False
        self.epsilon = epsilon

    def getDistribution(self, state):
        # ghost function

        dist = util.Counter()
        legalActions = state.getLegalActions(self.index)
        for a in legalActions:
            dist[a] = 0
        dist[self.getAction(state)] = 1
        dist.normalize()
        return dist

    def getAction(self, state):
        # pacman function

        legalActions = state.getLegalActions(self.index)

        #if we don't have learn yet, make random move + epsilon greedy
        if self.learning_algo is None or np.random.uniform() <= self.epsilon:
            move = legalActions[np.random.randint(0,len(legalActions))]
        else:
            move = legalActions[np.argmax(self.learning_algo.predict(np.array([(getDataState(state),a) for a in legalActions])))]

        if self.learn:
            self._saveOneStepTransistion(state,move)

        return move

    def startLearning(self):
        self.learn = True
        self.one_step_transistion = []

    def stopLearning(self):
        self.learn = False

    def learnFromPast(self):
        if len(self.one_step_transistion):
            if self.learning_algo is None:
#              self.learning_algo = MLPRegressor()
              #TODO: faire l'algo...
              pass


    def _saveOneStepTransistion(self,state,move):

        nextState = state.generateSuccessor(self.index,move)

        if self.index:
            #ghost reward
            reward = -util.manhattanDistance(nextState.getGhostPosition(self.index),
                                          nextState.getPacmanPosition()) - \
                     1000 * nextState.isWin() + 10000 * nextState.isLose()
        else:
            #pacman reward
            reward = -1 + 1000 * nextState.isWin() - 100000 * nextState.isLose()

        state_data = getDataState(state)
        nextState_data = getDataState(nextState)
        self.one_step_transistion.append((state_data,move,reward,nextState_data))

def getDataState(state):
  return (state.data.agentStates[:],state.getFood().copy(),state.getCapsules().copy())