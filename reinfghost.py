# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:14:05 2018

@author: Sarah
"""

from game import Actions
from util import manhattanDistance
import util
from ghostAgents import GhostAgent


class Reinfghost(GhostAgent):

    def __init__(self, index, prob_attack=1, prob_scaredFlee=1):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):

        return dist
