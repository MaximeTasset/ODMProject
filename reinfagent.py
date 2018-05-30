# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:12:23 2018

@author: Sarah and Maxime
"""
from pacman import Directions
from game import Agent
from ghostAgents import GhostAgent
from greedyghost import Greedyghost
from agentghost  import Agentghost
from brain import *
import util
import numpy as np
from copy import deepcopy
import sys
import tensorflow as tf
import scipy.signal
import gc
from heapq import nlargest
from queue import PriorityQueue

MAX_SIZE =  30

DIRECTION = [ Directions.NORTH,
              Directions.SOUTH,
              Directions.EAST,
              Directions.WEST,
              Directions.STOP]


class ReinfAgentFQI(GhostAgent,Agent):
    def __init__(self,index=0,round_training=5):
        self.one_step_transistions = []
        self.prev = None
        self.lastMove = 4
        self.index = index
        self.round_training = round_training
        self.learning_algo = None
        self.show = False
        self.learn = False
        if index:
            self.training_ghost = Greedyghost(index)
        else:
            self.training_pacman = Agentghost(index=0, time_eater=0, g_pattern=1)

    def get_History(self,reset=True):

        if reset:
          history = self.one_step_transistions
          self.one_step_transistions = []
        else:
          history = self.one_step_transistions.copy()
        return history

    def showLearn(self,show=True):
        self.show = show

    def startLearning(self):
        self.learn = True

    def stopLearning(self):
        self.learn = False

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
        legal = state.getLegalActions(self.index)
        if (self.round_training and not self.show) or self.learning_algo is None:
            if self.index:
                dist = self.training_ghost.getDistribution(state)
                dist.setdefault(0)
                dist_p = np.zeros(len(DIRECTION))
                for i,d in enumerate(DIRECTION):
                    dist_p[i] = dist[d]
                move = np.random.choice(dist_p,p=dist_p)
                move = DIRECTION[np.argmax(dist_p == move)]
            else:
                move = self.training_pacman.getAction(state)
        else:
            legalActions = list(map(DIRECTION.index,legal))
            state_data = tuple(getDataState(state,self.index).tolist())
            if np.random.uniform() > 0.1:
                a_dist = self.learning_algo.predict(np.array([state_data+(action,) for action in legalActions]))
                move = DIRECTION[legalActions[np.argmax(a_dist)]]
            else:
                move = np.random.randint(len(legalActions))
                move = DIRECTION[legalActions[move]]

        if not move in legal:
              possibleMoves = list(map(DIRECTION.index,legal))
              move = DIRECTION[np.random.choice(possibleMoves)]
        if self.learn:
            self._saveOneStepTransistion(state,move,False)
        return move

    def final(self,state):
        self._saveOneStepTransistion(state,None,True)
        self.lastMove = 4
        self.round_training -= 1

    def _saveOneStepTransistion(self,state,move,final):
        state_data = tuple(getDataState(state,self.index).tolist())
        if not self.prev is None:

            if self.index:
                # ghost reward:
                reward = -util.manhattanDistance(state.getGhostPosition(self.index),
                                              state.getPacmanPosition()) - \
                         100000 * state.isWin() + 100000 * state.isLose()
            else:
                # pacman reward:
                reward = -1 + 100000 * state.isWin() \
                        -100000 * state.isLose() + abs(state.getNumFood() - self.prev[0].getNumFood()) * 51 + \
                        (state.getPacmanPosition() in self.prev[0].getCapsules()) * 101

            possibleMoves = list(map(lambda x:(DIRECTION.index(x),),state.getLegalActions(self.index))) if not final else []
            self.one_step_transistions.append((self.prev[2],self.prev[1],reward,state_data,possibleMoves))

        if not final:
          move = DIRECTION.index(move)
          self.lastMove = move

          self.prev = (state.deepCopy(),(move,),state_data)
        else:
          self.prev = None


def computeFittedQIteration(samples,mlAlgo,N=400,gamma=.999):
    """
    " samples = [(state0,action0,reward0,state0',possibleMoveFromState0'),...,(stateN,actionN,rewardN,stateN',possibleMoveFromStateN')]
    " mlAlgo = an instance of the jean class
    "
    " Return: the training set for the Nth iteration of FQI
    """
    QnLSX = np.array([(s + a) for (s,a,_,_,_) in samples])
    QnLSY = np.array([r for (_,_,r,_,_) in samples])


    QN_it = deepcopy(mlAlgo)

    # For the first iteration:
    sys.stdout.write("\r \t{}/{}  ".format(1,N))
    sys.stdout.flush()
    QN_it.fit(QnLSX,QnLSY)

    # Creation of the array that will be used for predictions:
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
      sys.stdout.write("\r \t{}/{}  ".format(n+2,N))
      sys.stdout.flush()

      # One big call is much faster than multiple small ones:
      Qn_1 = QN_it.predict(topredict)
      # The recursion is used only when not in a terminal state:
      QnLSY = np.array([(gamma * max(Qn_1[index[s0,a0]]) if len(pos) else r) for (s0,a0,r,s1,pos) in samples])

      if n != N-2:
          QN_it.fit(QnLSX,QnLSY)

    return QnLSX,QnLSY


def discounted_return(x, gamma):
    # Used to calculate discounted_returned returns.
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def convertGridToNpArray(grid):
    array = np.zeros((grid.width,grid.height))
    for x in range(grid.width):
        for y in range(grid.height):
          array[x,y] = grid[x][y]
    return array


def distanceMap(grid,coord,maxPos=5):
  coord = tuple(coord)
  neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
  distance_map = np.empty_like(grid)
  distance_map.fill(sum(distance_map.shape))

  pq = PriorityQueue()
  pq.put((0,coord))
  while not pq.empty() and maxPos:
    dist,curr = pq.get()
    if distance_map[curr] > dist:
      distance_map[curr] = dist
      for x,y in neighbors:
        neighbor =  curr[0] + x, curr[1] + y

        if 0 <= neighbor[0] < grid.shape[0]:
          if 0 <= neighbor[1] < grid.shape[1]:
            if grid[tuple(neighbor)] >= 0:
              pq.put((dist+1,tuple(neighbor)))
              if grid[tuple(neighbor)]:
                maxPos -= 1

  return distance_map

def getDataState(state,index=0,maxPos=-1,vector=False):
    """
    " Returns a tuple whose first elements are the positions of all the agents,
    " and whose other elements contain the flattened food grid.
    """
    agent_pos = [st.getPosition() for st in state.data.agentStates]
    food_pos = state.getFood()
    walls_pos = state.getWalls()
    caps_pos = state.getCapsules()

    if vector:
      WALL = 0
      PACMAN = 1
      GHOST = 2
      FOOD = 3
      CAPS = 4
      data = np.zeros((walls_pos.width,walls_pos.height,5))
    else:
      data = np.zeros((walls_pos.width,walls_pos.height))
    foods = []
    if vector:
      for i,pos in enumerate(agent_pos):
          x,y = int(pos[0]),int(pos[1])
          if i:
              data[x,y,GHOST] = i
          else:
              data[x,y,PACMAN] = 1

    for i in range(walls_pos.width):
        for j in range(walls_pos.height):
            if not vector:
                if (i,j) in agent_pos:
                    index_agent = agent_pos.index((i,j))
                    data[i,j] = 10000 if not index_agent  else -index_agent*1000
                elif walls_pos[i][j]:
                    data[i,j] = -10
                elif food_pos[i][j] and not index:
                    foods.append((i,j))
                    data[i,j] = 2000
                elif (i,j) in caps_pos and not index:
                    foods.append((i,j))
                    data[i,j] = 2000
            else:
                if walls_pos[i][j]:
                    data[i,j,WALL] = 1
                if food_pos[i][j] and not index:
                    foods.append((i,j))
                    data[i,j,FOOD] = 1
                if (i,j) in caps_pos and not index:
                    foods.append((i,j))
                    data[i,j,CAPS] = 1

    if not index and maxPos != -1:
      dM = distanceMap(data,agent_pos[0],maxPos)
      for i,j in nlargest(len(foods)-maxPos,foods,key=lambda pt: dM[tuple(pt)]):
          if vector:
              data[i,j,FOOD] = 0
              data[i,j,CAPS] = 0
          else:
              data[i,j] = 0
    return data.flatten()

