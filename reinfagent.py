# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:12:23 2018

@author: Sarah and Maxime
"""
from pacman import Directions
from game import Agent
from game import Actions

from ghostAgents import GhostAgent

import util
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.base import clone
import sys

class ReinfAgent(GhostAgent,Agent):

    def __init__(self, index=0,epsilon=0.1):

        self.lastMove = Directions.STOP
        self.index = index

        self.learning_algo = None
        self.learn = False
        self.epsilon = epsilon
        self.prev = None
        self.one_step_transistions = []

    def getDistribution(self, state):
        # Ghost function

        dist = util.Counter()
        legalActions = state.getLegalActions(self.index)
        for a in legalActions:
            dist[a] = 0
        dist[self.getAction(state)] = 1
        dist.normalize()
        return dist

    def getAction(self, state):
        # Pacman function

        legalActions = state.getLegalActions(self.index)

        # If we don't have learn yet, make random move + epsilon greedy
        if self.learning_algo is None or np.random.uniform() <= self.epsilon:
            move = legalActions[np.random.randint(0,len(legalActions))]
            if Actions.directionToVector(move) == (0,0):
                move = legalActions[np.random.randint(0,len(legalActions))]
        else:
            move = legalActions[np.argmax(
                    self.learning_algo.predict(
                            np.array([(getDataState(state)+a) for a in map(Actions.directionToVector,legalActions)])))]

        if self.learn:
            self._saveOneStepTransistion(state,move)

        return move

    def startLearning(self):
        self.learn = True
        self.one_step_transistions = []

    def stopLearning(self):
        self.learn = False

    def learnFromPast(self,used_core=-1):
        if len(self.one_step_transistions):
            self.learning_algo = computeFittedQIteration(self.one_step_transistions,
                                                         N=60,
                                                         mlAlgo=ExtraTreesRegressor(n_estimators=100,n_jobs=used_core))

    def final(self,final_state):
      self._saveOneStepTransistion(final_state,None,True)

    def _saveOneStepTransistion(self,state,move,final=False):
        state_data = getDataState(state)
        if not self.prev is None:

            possibleMove = list(map(Actions.directionToVector,state.getLegalActions(self.index)))

            if self.index:
                #ghost reward
                reward = -util.manhattanDistance(state.getGhostPosition(self.index),
                                              state.getPacmanPosition()) - \
                         1000 * state.isWin() + 10000 * state.isLose()
            else:
                #pacman reward
                reward = -1 + 1000 * state.isWin() - \
                        100000 * state.isLose() + abs(state.getNumFood() + self.prev[0].getNumFood()) * 51 + \
                        (state.getPacmanPosition() in self.prev[0].getCapsules()) * 101

            self.one_step_transistions.append((state_data,self.prev[1],reward,self.prev[2],possibleMove))

        if not final:
          move = Actions.directionToVector(move)
          self.prev = (state.deepCopy(),move,state_data)
        else:
          self.prev = None



def computeFittedQIteration(samples,N=400,mlAlgo=ExtraTreesRegressor(n_estimators=100,n_jobs=-1),gamma=.95):
    """
    " samples = [(state0,action0,reward0,state1,possibleMoveFromState1),...,(stateN,actionN,rewardN,stateN+1,possibleMoveFromStateN+1)]
    " convergenceTestSet, None = no test set => return None
    "
    " Return: an trained instance of mlAlgo
    "
    " Note: this function assumes that an option like 'warm_start' is set to False or that a call to the fit function reset the model.

    """
    QnLSX = np.array([(s + a) for (s,a,_,_,_) in samples])
    QnLSY = np.array([r for (_,_,r,_,_) in samples])


    QN_it = clone(mlAlgo)

    # N=1
    QN_it.fit(QnLSX,QnLSY)


    # Creation of the array that will be used for predictions
    i = 0
    topredict = []
    index = {}
    for (s0,a0,r,s1,actionSpace) in samples:
      index[s0,a0] = []
      for a in actionSpace:
        topredict.append((s1+a))
        index[s0,a0].append(i)
        i +=1

    topredict = np.array(topredict)

    for n in range(0,N-1):
      sys.stdout.write("\r{}/{}  ".format(n+2,N))
      sys.stdout.flush()

      # One big call is much faster than multiple small ones.
      Qn_1 = QN_it.predict(topredict)
      # The recursion is used only when not in a terminal state
      QnLSY = np.array([(gamma * max(Qn_1[index[s0,a0]]) if abs(r) < 1000 else r) for (s0,a0,r,s1,_) in samples])


      QN_it.fit(QnLSX,QnLSY)

    return QN_it

def convertGridToNpArray(grid):
    array = np.zeros((grid.width,grid.height))
    for x in range(grid.width):
        for y in range(grid.height):
          array[x,y] = grid[x][y]
    return array

def getDataState(state):
    """
    " Returns a tuple whose first elements are the positions of all the agents,
    " and whose other elements contain the flattened food grid.
    """
    #,state.getCapsules().copy()

    return tuple(np.array(
            [st.getPosition() for st in state.data.agentStates]).flatten().tolist() \
            + convertGridToNpArray(state.getFood()).flatten().tolist())