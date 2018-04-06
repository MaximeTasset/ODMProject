# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:38:41 2018

@author: Sarah
"""
import pacman
from reinfpacman import Reinfpacman
from reinfghost import Reinfghost

def iterativeA3c(nb_ghost=3):
    
    agents = list()
    
    agent.append(Reinfpacman()) 
    
    for i in range(1,nb_ghosts+1):
        agent.append(Reinfghost(i))
    
    while nb_it<100 or consec_wins<50:
        pass