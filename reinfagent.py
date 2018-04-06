# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:12:23 2018

@author: Sarah
"""
from pacman import Directions
from game import Agent
from ghostAgents import GhostAgent
import util
import sklearn

class ReinfAgent(GhostAgent,Agent):

    def __init__(self, index=0, time_eater=40, g_pattern=-1, 
                 prob_attack=1, prob_scaredFlee=1):

        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee
        self.learning_algo = sklearn.MLPRegressor()
    
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
        move = legalActions[1]
        return move
    
    def startLearning(self):
        learn = True
        
    def stopLearning(self):
        learn = False
