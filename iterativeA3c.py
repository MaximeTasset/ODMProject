# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:38:41 2018

@author: Sarah
"""
import pacman
from reinfagent import ReinfAgent
import layout
import sys

def iterativeA3c(nb_ghosts=3,nb_training=10,display_mode='graphics',round_training=5):

    # Choose a display format
    if display_mode == 'quiet':
        import textDisplay
        display = textDisplay.NullGraphics()
    elif display_mode == 'text':
        import textDisplay
        textDisplay.SLEEP_TIME = 0.1
        display = textDisplay.PacmanGraphics()
    else:
        import graphicsDisplay
        display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1)

    agents = list()

    agents.append(ReinfAgent())

    for i in range(1,nb_ghosts+1):
        agents.append(ReinfAgent(i))

    layout_instance = layout.getLayout('mediumClassic')

    nb_it = 0
    consec_wins = 0

    while nb_it<100 or abs(consec_wins)<50:

        for i in range(nb_ghosts+1):
            for j in range(round_training+1):
                sys.stdout.write("\r                       {}/{}       ".format(j+1,round_training))
                sys.stdout.flush()
                agents[i].startLearning()
                if j != round_training:
                    games = pacman.runGames(layout_instance,agents[0],agents[1:],display,nb_training,False,timeout=30,numTraining=nb_training)
                    #compute how many consecutive win ghosts or pacman have
                    for game in games:
                        if game.state.isWin():
                            consec_wins = max(1,consec_wins+1)
                        else:
                            consec_wins = min(-1,consec_wins-1)
                else:
                    games = pacman.runGames(layout_instance,agents[0],agents[1:],display,1,False,timeout=30)

                agents[i].learnFromPast()
                agents[i].stopLearning()
        nb_it += 1


    return agents