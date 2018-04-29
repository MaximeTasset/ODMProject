# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:12:23 2018

@author: Sarah and Maxime
"""
from pacman import Directions
from game import Agent
from game import Actions

from ghostAgents import GhostAgent

from brain import *

import util
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.base import clone
import sys
import tensorflow as tf
import scipy.signal

DIRECTION = { 0 : Directions.NORTH,
              1 : Directions.SOUTH,
              2 : Directions.EAST,
              3 : Directions.WEST,
              4 : Directions.STOP}

def dF(val):
    return 0.005

class ReinfAgent(GhostAgent,Agent):

    def __init__(self,optim, global_episodes,sess,s_size,a_size,grid_size, index=0,
                 name="worker",global_scope='global',epsilon=1,gamma=0.95,min_epsilon=0.01):

        self.lastMove = Directions.STOP
        self.index = index
        self.gamma = gamma

        self.learning_algo = None
        self.learn = False
        self.epsilon = epsilon
        self.prev = None
        self.one_step_transistions = []
        self.opt = tf.train.AdamOptimizer()

        self.name = name
        self.global_scope = global_scope

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

    def diminueEpsilon(self):
        if self.min_epsilon != self.epsilon:
            self.epsilon = max(self.min_epsilon,self.epsilon-dF(self.epsilon))

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
            s = getDataState(state)
            # If we don't have learn yet or epsilon greedy, make random move
            if np.random.uniform() <= self.epsilon:
                v = self.sess.run(self.local_AC.value,
                                feed_dict={self.local_AC.inputs:[s],
                                self.local_AC.state_in[0]:self.rnn_state[0],
                                self.local_AC.state_in[1]:self.rnn_state[1]})
                move = legalActions[np.random.randint(0,len(legalActions))]
                if Actions.directionToVector(move) == (0,0):
                    move = legalActions[np.random.randint(0,len(legalActions))]
            else:

                a_dist,v,self.rnn_state = self.sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                                    feed_dict={self.local_AC.inputs:[s],
                                    self.local_AC.state_in[0]:self.rnn_state[0],
                                    self.local_AC.state_in[1]:self.rnn_state[1]})

                #move = np.random.choice(a_dist[0],p=a_dist[0])
                #move = DIRECTION[np.argmax(a_dist == move)]

                #we remove the illegal move from the distribution
                a_dist = [a if DIRECTION[i] in legalActions else 0 for i,a in enumerate(a_dist[0].tolist())]
                a_dist = np.array(a_dist)
                a_dist = a_dist / sum(a_dist)
                move = np.random.choice(a_dist,p=a_dist)
                move = DIRECTION[np.argmax(a_dist == move)]
#                sorted_probas = sorted(a_dist)
#                i = 0

#                move = DIRECTION[a_dist.index(sorted_probas[i])]
#                print(legalActions)
#                print('CACA',a_dist,sorted_probas)
#                a = []
#                while not move in legalActions:
#                    move = np.random.choice(a_dist[0],p=a_dist[0])
#                    move = DIRECTION[np.argmax(a_dist == move)]
#                    i += 1
                    #print(move,i)
#                    try:
#                        a.append(move)
#                        move = DIRECTION[a_dist.index(sorted_probas[i])]
#                    except IndexError:
#                        print(a_dist,sorted_probas,i,legalActions,a)
#                        raise IndexError
#                    if i == 100:
#                        move = legalActions[np.random.randint(0,len(legalActions))]
#                        if Actions.directionToVector(move) == (0,0):
#                            move = legalActions[np.random.randint(0,len(legalActions))]


            if self.learn:
                self._saveOneStepTransistion(state,move,False,v[0,0])

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
      with self.sess.as_default(), self.sess.graph.as_default():
          self._saveOneStepTransistion(final_state,None,True)

    def _saveOneStepTransistion(self,state,move,final,v=None):
        state_data = getDataState(state)
        if not self.prev is None:

#            possibleMove = list(map(Actions.directionToVector,state.getLegalActions(self.index)))

            if self.index:
                #ghost reward
                reward = -util.manhattanDistance(state.getGhostPosition(self.index),
                                              state.getPacmanPosition()) - \
                         1000 * state.isWin() + 10000 * state.isLose()
            else:
                #pacman reward
                reward = -1 + 1000 * state.isWin() \
                        -100000 * state.isLose() + abs(state.getNumFood() + self.prev[0].getNumFood()) * 51 + \
                        (state.getPacmanPosition() in self.prev[0].getCapsules()) * 101

            self.one_step_transistions.append([state_data,self.prev[1],reward,self.prev[2],self.prev[3]])
        if len(self.one_step_transistions) == 30 or final:
            self.diminueEpsilon()
            v1 = self.sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[state_data],
                            self.local_AC.state_in[0]:self.rnn_state[0],
                            self.local_AC.state_in[1]:self.rnn_state[1]})[0,0]
            self.train(self.one_step_transistions,self.sess,self.gamma,v1)
            self.one_step_transistions = []

        if not final:
          for i,m in DIRECTION.items():
            if m == move:
              move = i
              break
          self.prev = (state.deepCopy(),move,state_data,v)
        else:
          self.prev = None

    def train(self,rollout,sess,gamma,bootstrap_value):
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

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

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
    agent_pos = [st.getPosition() for st in state.data.agentStates]
    food_pos = state.getFood()
    walls_pos = state.getWalls()
    caps_pos = state.getCapsules()
    nb_agent = len(agent_pos)

    data = np.zeros((walls_pos.width,walls_pos.height))
    for i in range(walls_pos.width):
        for j in range(walls_pos.height):
            if (i,j) in agent_pos:
                data[i,j] = agent_pos.index((i,j)) + 1
            elif walls_pos[i][j]:
                data[i,j] = nb_agent + 1
            elif food_pos[i][j]:
                data[i,j] = nb_agent + 2
            elif (i,j) in caps_pos:
                data[i,j] = nb_agent + 3
    return data.flatten()

