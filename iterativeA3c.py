# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:38:41 2018

@author: Sarah and Maxime
"""
import pacman
from reinfagent import ReinfAgent,getDataState
import layout

import tensorflow as tf
from brain import *

import sys
from copy import deepcopy
from multiprocessing.pool import ThreadPool
#from multiprocessing import Pool
import psutil

def runGames(kargs):
    return pacman.runGames(**kargs)

def iterativeA3c(nb_ghosts=3,nb_training=20,display_mode='graphics',
                 round_training=5,num_parallel=1,nb_cores=-1):

    pool = ThreadPool(num_parallel)
#    pool = Pool(num_parallel)

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


    layout_instance = layout.getLayout('mediumClassic')

    initState = pacman.GameState()
    initState.initialize(layout_instance, nb_ghosts)
    s_size = len(getDataState(initState))

    master_networks = [AC_Network(s_size,4 if i else 5,"global_"+str(i),None,global_scope="global_"+str(i))  for i in range(0,nb_ghosts+1)]# Generate global network
    global_episodes = [tf.Variable(0,dtype=tf.int32,name='global_episodes'+str(i),trainable=False)  for i in range(0,nb_ghosts+1)]
    optims = [tf.train.AdamOptimizer(learning_rate=1e-4) for i in range(0,nb_ghosts+1)]
    parallel_agents = [[ReinfAgent(optim=optims[i],global_episodes=global_episodes[i],index=i,name="worker_{}_{}".format(i,j),global_scope="global_"+str(i)) for i in range(0,nb_ghosts+1)] for j in range(num_parallel)]
    main_agents = parallel_agents[0]

    args = [{"layout":layout_instance,
             "pacman":parallel_agents[i][0],
             "ghosts":parallel_agents[i][1:],
             "display":display,
             "numGames":nb_training,
             "record":False,
             "numTraining":nb_training,
             "timeout":30} for i in range(num_parallel)]

    nb_it = 0
    consec_wins = 0

    while nb_it<100 or abs(consec_wins)<50:

        for i in range(nb_ghosts+1):
            for j in range(round_training+1):
#                if j != round_training:
#                    sys.stdout.write("\r           {}/{}       ".format(j+1,round_training))
#                else:
#                    sys.stdout.write("\r           Final result       ")
#                sys.stdout.flush()
                for agents in parallel_agents:
                  agents[i].startLearning()

                if j != round_training:
                    pool.map(runGames,args)
                    for k in range(1,num_parallel):
                        main_agents[i].one_step_transistions.extend(parallel_agents[k][i].one_step_transistions)
                    main_agents[i].learnFromPast(nb_cores)
                    for k in range(1,num_parallel):
                        parallel_agents[k][i].learning_algo = deepcopy(main_agents[i].learning_algo)
                else:
                    games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,False,timeout=30)
                    # Compute how many consecutive wins ghosts or pacman have
                    for game in games:
                        if game.state.isWin():
                            consec_wins = max(1,consec_wins+1)
                        else:
                            consec_wins = min(-1,consec_wins-1)


                for agents in parallel_agents:
                  agents[i].stopLearning()

        nb_it += 1


    return master_networks

if __name__ is "__main__":
  iterativeA3c(nb_ghosts=3,nb_training=20,display_mode='graphics',
               round_training=5,num_parallel=1,
               nb_cores=max(1,psutil.cpu_count()-1))
#  max(1,psutil.cpu_count())