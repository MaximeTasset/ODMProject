# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:38:41 2018

@author: Sarah
"""
import pacman
from reinfagent import ReinfAgent
import layout
import sys
from copy import deepcopy
from multiprocessing.pool import ThreadPool

def iterativeA3c(nb_ghosts=3,nb_training=10,display_mode='graphics',round_training=5,num_parallel=20):

    pool = ThreadPool(num_parallel)
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


    parallel_agents = [[ReinfAgent(i) for i in range(0,nb_ghosts+1)] for _ in range(num_parallel)]
    main_agents = parallel_agents[0]
    layout_instance = layout.getLayout('mediumClassic')

    nb_it = 0
    consec_wins = 0

    while nb_it<100 or abs(consec_wins)<50:

        for i in range(nb_ghosts+1):
            for j in range(round_training+1):
                if j != round_training:
                    sys.stdout.write("\r                       {}/{}       ".format(j+1,round_training))
                else:
                    sys.stdout.write("\r                       Final result       ")
                sys.stdout.flush()
                for agents in parallel_agents:
                  agents[i].startLearning()

                if j != round_training:
                    pool.map(lambda agents : pacman.runGames(layout_instance,
                                                              agents[0],
                                                              agents[1:],
                                                              display,
                                                              nb_training,
                                                              False,
                                                              timeout=30,
                                                              numTraining=nb_training),parallel_agents)
                    for k in range(1,num_parallel):
                        main_agents[i].one_step_transistions.extend(parallel_agents[k][i].one_step_transistions)
                else:
                    games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,False,timeout=30)
                     #compute how many consecutive win ghosts or pacman have
                    for game in games:
                        if game.state.isWin():
                            consec_wins = max(1,consec_wins+1)
                        else:
                            consec_wins = min(-1,consec_wins-1)

                main_agents[i].learnFromPast()
                for k in range(1,num_parallel):
                    parallel_agents[k][i].learning_algo = deepcopy(main_agents[i].learning_algo)
                for agents in parallel_agents:
                  agents[i].stopLearning()

        nb_it += 1


    return main_agents