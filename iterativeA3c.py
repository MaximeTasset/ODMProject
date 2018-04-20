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
import os
import imageio as io

import graphicsDisplay

def runGames(kargs):
    return pacman.runGames(**kargs)

def iterativeA3c(nb_ghosts=3,display_mode='graphics',
                 round_training=5,num_parallel=1,nb_cores=-1):

    tf.reset_default_graph()
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
    nb_ghosts = min(len(layout_instance.agentPositions)-1,nb_ghosts)

    # Get the length of a state:
    initState = pacman.GameState()
    initState.initialize(layout_instance, nb_ghosts)
    s_size = len(getDataState(initState))
    grid_size = initState.getWalls().width,initState.getWalls().height
    # Generate global network
    master_networks = [AC_Network(s_size,4 if i else 5,grid_size,"global_"+str(i),None,
                                  global_scope="global_"+str(i))
                                    for i in range(0,nb_ghosts+1)]
    global_episodes = [tf.Variable(0,dtype=tf.int32,name='global_episodes'+str(i),trainable=False) for i in range(0,nb_ghosts+1)]
    optims = [tf.train.AdamOptimizer(learning_rate=1e-4) for i in range(0,nb_ghosts+1)]


    with tf.Session() as sess:

        parallel_agents = [[ReinfAgent(optims[i],global_episodes[i],sess,
                                       s_size,4 if i else 5,grid_size,index=i,
                                       name="worker_{}_{}".format(i,j),
                                       global_scope="global_"+str(i))
                                        for i in range(0,nb_ghosts+1)]
                                        for j in range(num_parallel)]

        sess.run(tf.global_variables_initializer())
        main_agents = parallel_agents[0]

        args = [{"layout":layout_instance,
                 "pacman":parallel_agents[i][0],
                 "ghosts":parallel_agents[i][1:],
                 "display":display,
                 "numGames":1,
                 "record":False,
                 "numTraining":1,
                 "timeout":30} for i in range(num_parallel)]

        nb_it = 0
        consec_wins = 0

        while nb_it<100 or abs(consec_wins)<50:

            for i in range(nb_ghosts+1):
                print("Pacman" if not i else "Ghost {}".format(i))
                for agents in parallel_agents:
                    agents[i].startLearning()
                curr_round_training = round_training if i else max(round_training,round_training*nb_ghosts)
                for j in range(curr_round_training):
                    sys.stdout.write("\r                {}/{}       ".format(j+1,curr_round_training))
                    sys.stdout.flush()

                    pool.map(runGames,args)

                for agents in parallel_agents:
                    agents[i].stopLearning()

                sys.stdout.write("\r           Final result       \n")
                sys.stdout.flush()
                games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,False,timeout=30)
                # Compute how many consecutive wins ghosts or pacman have
                # consec_wins is negative if the ghosts have won, positive otherwise.
                # abs(consec_wins) is the number of consecutive wins.
                for game in games:
                    if game.state.isWin():
                        consec_wins = max(1,consec_wins+1)
                    else:
                        consec_wins = min(-1,consec_wins-1)

                make_gif('agent_{}_nbrounds_{}.mp4'.format(i,nb_it))
                graphicsDisplay.FRAME_NUMBER = 0
            nb_it += 1


    return master_networks


def make_gif(filename='movie.mp4'):
    filename = 'videos/'+filename
    # The images to use are in subfolder frames of the current folder:
    nb_frames = len(os.listdir(os.getcwd()+"/frames"))

    os.makedirs('videos',exist_ok=True)

    with io.get_writer(filename, mode='I',macro_block_size=None) as writer:
        for i in range(nb_frames):
            filename = 'frames/frame_%08d.ps' % i
            image = io.imread(filename)
            writer.append_data(image)
            os.remove(filename)


if __name__ is "__main__":
  iterativeA3c(nb_ghosts=1,round_training=500,display_mode='graphics',num_parallel=psutil.cpu_count(),
               nb_cores=max(1,psutil.cpu_count()-1))
#  max(1,psutil.cpu_count())