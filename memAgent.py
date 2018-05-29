# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:55:17 2018

@author: Maxime
"""

from reinfagent import DIRECTION,MAX_SIZE,getDataState
from ghostAgents import GhostAgent
from greedyghost import Greedyghost
from agentghost  import Agentghost
import util
import layout

import pacman

import textDisplay

from game import Agent
import numpy as np

from threading import Thread
from multiprocessing import Queue, Process,Pool,Manager
from multiprocessing.queues import Empty

import os

from pickle import dump

from time import sleep

TRAIN_TRIGGER = []

class MemAgent(GhostAgent,Agent):
  def __init__(self,queue,q_index,index=0,vector=True,epsilon=0.5):
      self.index = index
      self.q_index = q_index
      self.queue = queue
      self.prev = None
      self.vector = vector
      if index:
          self.training_ghost = Greedyghost(index)
      else:
          self.training_pacman = Agentghost(index=0, time_eater=0, g_pattern=1)
      self.epsilon = epsilon
      self.count = 0

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
      if self.index:
          dist = self.training_ghost.getDistribution(state)
          dist.setdefault(0)
          dist_p = np.zeros(len(DIRECTION))
          for i,d in enumerate(DIRECTION):
              dist_p[i] = dist[d]
          move = np.random.choice(dist_p,p=dist_p)
          move = DIRECTION[np.argmax(dist_p == move)]
      else:
          legal = state.getLegalActions(self.index)
          if self.epsilon > np.random.uniform():
            move = self.training_pacman.getAction(state)
            if not move in legal:
              possibleMoves = list(map(DIRECTION.index,legal))
              move = DIRECTION[np.random.choice(possibleMoves)]
          else:
            possibleMoves = list(map(DIRECTION.index,legal))
            move = DIRECTION[np.random.choice(possibleMoves)]

      self._saveOneStepTransistion(state,move,False)
      return move

  def final(self,state):
        self._saveOneStepTransistion(state,None,True)

  def _saveOneStepTransistion(self,state,move,final):
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
            possibleMoves = list(map(lambda x:(DIRECTION.index(x),),state.getLegalActions(self.index))) if not final else []
            self.queue.put((self.index,self.q_index,[self.prev[2],self.prev[1],reward,state_data,possibleMoves]))
            self.count += 1

        if self.count == MAX_SIZE or final:
            self.queue.put((self.index,self.q_index,TRAIN_TRIGGER))
            self.count = 0

        if not final:
          move = DIRECTION.index(move)
          self.prev = (state.deepCopy(),move,state_data)
        else:
          self.prev = None


class Player:
    def __init__(self,rounds,**kargs):
        self.rounds = rounds
        self.kargs = kargs
    def play(self):
        for i in range(self.rounds):
            pacman.runGames(**self.kargs)

def runGames(kargs):
    Player(**kargs).play()
    return

class Save(Thread):
    def __init__(self,queue,nb_ghosts,folder):
        super().__init__()
        self.queue = queue
        self.stop = False
        self.nb_ghosts = nb_ghosts
        self.folder = folder

    def stop_me(self):
        self.stop = True

    def run(self):
        os.makedirs(self.folder,exist_ok=True)
        current_folder =  os.path.join(self.folder,str(self.nb_ghosts))
        os.makedirs(current_folder,exist_ok=True)
        agent_folders = [os.path.join(current_folder,str(i)) for i in range(0,self.nb_ghosts+1)]
        for f in agent_folders:
            os.makedirs(f,exist_ok=True)
        agent_counters = np.empty(self.nb_ghosts+1)
        for i in range(self.nb_ghosts+1):
            agent_counters[i] = len(os.listdir(agent_folders[i]))
        agent_lists = [{} for i in range(0,self.nb_ghosts+1)]

        while not (self.queue.empty() and self.stop):
            try:
                index,q_index,value = self.queue.get_nowait()
                if value == TRAIN_TRIGGER:

                    if q_index in agent_lists[index]:
                        with open(os.path.join(agent_folders[index],str(int(agent_counters[index]))+'.save'),'wb') as f:
                            dump(agent_lists[index][q_index],f)
                        del agent_lists[index][q_index]
                        agent_counters[index] += 1
                else:
                    try:
                        agent_lists[index][q_index].append(value)
                    except KeyError:
                        agent_lists[index][q_index] = [value]

            except Empty:
                sleep(1)



def main(nb_ghosts=3,rounds=100,num_parallel=4,nb_cores=4, folder='videos',layer='mediumClassic',vector=True,epsilon=.2):

    display = textDisplay.NullGraphics()

    layout_instance = layout.getLayout(layer)
    nb_ghosts = min(len(layout_instance.agentPositions)-1,nb_ghosts)
    m = Manager()
    queue = m.Queue()

    parallel_agents = [[MemAgent(queue,j,i,vector,epsilon) for i in range(0,nb_ghosts+1)] for j in range(num_parallel)]


    args = [{"rounds":rounds,
             "layout":layout_instance,
             "pacman":parallel_agents[i][0],
             "ghosts":parallel_agents[i][1:],
             "display":display,
             "numGames":1,
             "record":False,
             "numTraining":1,
             "timeout":30} for i in range(num_parallel)]

    pool = Pool(nb_cores)
#    process = [Process(target=runGames, args=(args[i],)) for i in range(num_parallel)]
#    for p in process:
#        p.start()
#
#    for p in process:
#        p.join()
    player = Save(queue,nb_ghosts,folder)
    player.start()

    pool.map(runGames,args)

    player.stop_me()
    player.join()

    pool.close()
#    os.makedirs(folder,exist_ok=True)
#    agent_folders = [os.path.join(folder,str(i)) for i in range(0,nb_ghosts+1)]
#    for f in agent_folders:
#        os.makedirs(f,exist_ok=True)
#    agent_counters = np.zeros(nb_ghosts+1)
#    agent_lists = [{} for i in range(0,nb_ghosts+1)]
#
#    while not queue.empty():
#        index,q_index,value = queue.get()
#        if value == TRAIN_TRIGGER:
#
#            if q_index in agent_lists[index]:
#                with open(os.path.join(agent_folders[index],str(int(agent_counters[index]))+'.save'),'wb') as f:
#                    dump(agent_lists[index][q_index],f)
#                del agent_lists[index][q_index]
#                agent_counters[index] += 1
#        else:
#            try:
#                agent_lists[index][q_index].append(value)
#            except KeyError:
#                agent_lists[index][q_index] = [value]
if __name__ == '__main__':
    main(nb_ghosts=1,rounds=2400,num_parallel=4,nb_cores=4, folder='games',layer='mediumClassic',vector=True,epsilon=.2)
#    main(nb_ghosts=0,rounds=100,num_parallel=4,nb_cores=4, folder='games',layer='mediumClassic',vector=True,epsilon=.2)
#    main(nb_ghosts=3,rounds=100,num_parallel=4,nb_cores=4, folder='games',layer='mediumClassic',vector=True,epsilon=.2)
