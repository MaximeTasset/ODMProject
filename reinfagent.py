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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
#from sklearn.base import clone
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


def dF(val):
    return 0.00

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
            legalActions = list(map(DIRECTION.index,state.getLegalActions(self.index)))
            state_data = tuple(getDataState(state,self.index).tolist())
            if np.random.uniform() > 0.1:
                a_dist = self.learning_algo.predict(np.array([state_data+(action,) for action in legalActions]))
                move = DIRECTION[legalActions[np.argmax(a_dist)]]
            else:
                move = np.random.randint(len(legalActions))
                move = DIRECTION[legalActions[move]]

#            if not self.index:
#                a_dist[DIRECTION.index(Directions.REVERSE[DIRECTION[self.lastMove]])] /= 4
#                a_dist = a_dist / sum(a_dist)




#        legalActions = state.getLegalActions(self.index)
#        s = getDataState(state)
        if self.learn:
            self._saveOneStepTransistion(state,move,False)
        return move

    def final(self,state):
        self._saveOneStepTransistion(state,None,True)
        self.lastMove = 4

    def _saveOneStepTransistion(self,state,move,final):
        state_data = tuple(getDataState(state,self.index).tolist())
        if not self.prev is None:

            if self.index:
                #ghost reward
                reward = -util.manhattanDistance(state.getGhostPosition(self.index),
                                              state.getPacmanPosition()) - \
                         100000 * state.isWin() + 100000 * state.isLose()
            else:
                #pacman reward
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

class ReinfAgent(GhostAgent,Agent):

    def __init__(self,optim, global_episodes,sess,s_size,a_size,grid_size, index=0,
                 name="worker",global_scope='global',round_training=5,gamma=0.999,
                 epsilon=1,min_epsilon=0.01,vector=False):

        self.lastMove = 4
        self.index = index
        self.gamma = gamma

        self.vector = vector

        self.learning_algo = None
        self.learn = False
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.round_training = round_training
        self.prev = None
        self.one_step_transistions = []
        self.opt = tf.train.AdamOptimizer()

        self.name = name
        self.global_scope = global_scope
#        with open(self.name+'.txt','w'):
#            pass
        self.optim = optim
        self.global_episodes = global_episodes
        self.sess = sess
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.index))
        self.local_AC = AC_Network(s_size,a_size,grid_size,self.name,self.optim,global_scope=self.global_scope)
        self.update_local_ops = update_target_graph(self.global_scope,self.name)
        self.rnn_state = self.local_AC.state_init
        self.batch_rnn_state = self.rnn_state
        if index:
            self.training_ghost = Greedyghost(index)
        else:
            self.training_pacman = Agentghost(index=0, time_eater=0, g_pattern=1)
        self.count = 0
        self.show = False
#        self.nb_move = 0

    def diminueEpsilon(self):
        if self.min_epsilon != self.epsilon and self.count == 20:
            self.epsilon = max(self.min_epsilon,self.epsilon-dF(self.epsilon))
            self.count = 0
        self.count += 1
    def registerInitialState(self,state):
        self.sess.run(self.update_local_ops)

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
        with self.sess.as_default(), self.sess.graph.as_default():

            legalActions = state.getLegalActions(self.index)
            s = getDataState(state,self.index,vector=self.vector)
            # If we learn and epsilon greedy, make random move
            if self.round_training and not self.show:
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

#            if self.learn and np.random.uniform() <= self.epsilon:
                a_dist,v,self.rnn_state = self.sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                                    feed_dict={self.local_AC.inputs:[s],
                                    self.local_AC.state_in[0]:self.rnn_state[0],
                                    self.local_AC.state_in[1]:self.rnn_state[1]})
#                move = legalActions[np.random.randint(0,len(legalActions))]

#                if Actions.directionToVector(move) == (0,0):
#                    move = legalActions[np.random.randint(0,len(legalActions))]
            else:

                a_dist,v,self.rnn_state = self.sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                                    feed_dict={self.local_AC.inputs:[s],
                                    self.local_AC.state_in[0]:self.rnn_state[0],
                                    self.local_AC.state_in[1]:self.rnn_state[1]})

#                re = a_dist
                #we remove the illegal move from the distribution
                a_dist = [a if DIRECTION[i] in legalActions else 0 for i,a in enumerate(a_dist[0].tolist())]
                a_dist = np.array(a_dist)

                #we don't want to go in the opposite direction


#                with open(self.name+'.txt','a') as f:
#                    f.write('{}:{}\n'.format([st.getPosition() for st in state.data.agentStates],a_dist))

               # with open(self.name+'.txt','a') as f:
              #      f.write("{}:{}:{}:{}\n".format(re,prev,a_dist,self.epsilon))
                if np.random.uniform() <= self.epsilon:
                    if not self.index:
                        a_dist[DIRECTION.index(Directions.REVERSE[DIRECTION[self.lastMove]])] /= 4
#                        prev = a_dist.copy()
                        a_dist = a_dist / sum(a_dist)
                    move = np.random.choice(a_dist,p=a_dist)
                    move = DIRECTION[np.argmax(a_dist == move)]
                else:
                    move = DIRECTION[np.argmax(a_dist)]

            for i,m in enumerate(DIRECTION):
                if m == move:
                  self.lastMove = i
                  break
#            if self.nb_move == 100:
#                gc.collect()
#                self.nb_move = 1
#            else:
#                self.nb_move += 1

            if self.learn:
                self._saveOneStepTransistion(state,move,False,v[0,0])

            return move
    def showLearn(self,show=True):
        self.show = show


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
            self.one_step_transistions.clear()


    def final(self,final_state):
#      with self.sess.as_default(), self.sess.graph.as_default():
          self._saveOneStepTransistion(final_state,None,True)
          self.lastMove = 4
          gc.collect()
          if self.learn and self.round_training:
            self.round_training -= 1

    def _saveOneStepTransistion(self,state,move,final,v=None):
        state_data = getDataState(state,self.index,vector=self.vector)
        if not self.prev is None:

#            possibleMove = list(map(Actions.directionToVector,state.getLegalActions(self.index)))

            if self.index:
                #ghost reward
                reward = -util.manhattanDistance(state.getGhostPosition(self.index),
                                              state.getPacmanPosition()) - \
                         100000 * state.isWin() + 100000 * state.isLose()
            else:
                #pacman reward
                reward = -1 + 100000 * state.isWin() \
                        -100000 * state.isLose() + abs(state.getNumFood() - self.prev[0].getNumFood()) * 51 + \
                        (state.getPacmanPosition() in self.prev[0].getCapsules()) * 101
#                if self.name.endswith("0"):
#                    with open(self.name+'.txt','a') as f:
#                        f.write('from {} to {}: {}\n'.format(state.getPacmanPosition(),self.prev[0].getPacmanPosition(),reward))

            self.one_step_transistions.append([self.prev[2],self.prev[1],reward,state_data,self.prev[3]])

        if len(self.one_step_transistions) == MAX_SIZE or final:
            if not self.round_training:
                self.diminueEpsilon()
            v1 = self.sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[state_data],
                            self.local_AC.state_in[0]:self.rnn_state[0],
                            self.local_AC.state_in[1]:self.rnn_state[1]})[0,0]
            self.train(self.one_step_transistions,self.sess,self.gamma,v1)
            self.one_step_transistions = []
            self.sess.run(self.update_local_ops)

        if not final:
          move = DIRECTION.index(move)
          self.lastMove = move
          self.prev = (state.deepCopy(),move,state_data,v)
        else:
          self.prev = None

    def train(self,rollout,sess,gamma,bootstrap_value):
        with self.sess.as_default(), self.sess.graph.as_default():
            if not len(rollout):
                return
            rollout = np.array(rollout)
            observations = rollout[:,0]
            actions = rollout[:,1]
            rewards = rollout[:,2]
            next_observations = rollout[:,3]
            values = rollout[:,4]

            # Here we take the rewards and values from the rollout, and use them to
            # generate the advantage and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
            discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
            self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
            advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
            advantages = discount(advantages,gamma)

            # Update the global network using gradients from loss
            # Generate network statistics to periodically save
            feed_dict = {self.local_AC.target_v:discounted_rewards,
                self.local_AC.inputs:np.vstack(observations),
                self.local_AC.actions:actions,
                self.local_AC.advantages:advantages,
                self.local_AC.state_in[0]:self.batch_rnn_state[0],
                self.local_AC.state_in[1]:self.batch_rnn_state[1]}
            v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.state_out,
                self.local_AC.apply_grads],
                feed_dict=feed_dict)


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

    # N=1
    sys.stdout.write("\r \t{}/{}  ".format(1,N))
    sys.stdout.flush()
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
      sys.stdout.write("\r \t{}/{}  ".format(n+2,N))
      sys.stdout.flush()

      # One big call is much faster than multiple small ones.
      Qn_1 = QN_it.predict(topredict)
      # The recursion is used only when not in a terminal state
      QnLSY = np.array([(gamma * max(Qn_1[index[s0,a0]]) if len(pos) else r) for (s0,a0,r,s1,pos) in samples])

      if n != N-2:
          QN_it.fit(QnLSX,QnLSY)

    return QnLSX,QnLSY #QN_it

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
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
    #,state.getCapsules().copy()
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
          data[i,j] = 0
    return data.flatten()

