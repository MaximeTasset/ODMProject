
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:38:41 2018

@author: Sarah and Maxime
"""
import pacman
from reinfagent import ReinfAgentFQI,computeFittedQIteration
import layout
import numpy as np
import sys
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import os
import imageio as io
import gc
from copy import deepcopy
from sklearn.ensemble import ExtraTreesRegressor
from pickle import load


def runGamesFQI(kargs):
    games = pacman.runGames(**kargs)
    return (games,kargs)

def iterativeFQI(nb_ghosts=3,display_mode='graphics',
                 round_training=5,rounds=100,num_parallel=1,nb_cores=-1,
                 folder='videos',loadFrom='games',layer='mediumClassic'):

    pool = Pool(nb_cores)

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


    layout_instance = layout.getLayout(layer)
    nb_ghosts = min(len(layout_instance.agentPositions)-1,nb_ghosts)

    # Generate global network
    master_networks = [ExtraTreesRegressor(n_estimators=10,n_jobs=nb_cores,warm_start=True) for i in range(0,nb_ghosts+1)]

    factor_pacman = 2

    parallel_agents = [[ReinfAgentFQI(index=i,
                                      round_training=round_training if i else max(round_training,factor_pacman*round_training*nb_ghosts))
                                    for i in range(0,nb_ghosts+1)]
                                    for j in range(num_parallel)]

    current_folder = os.path.join(loadFrom,str(nb_ghosts))
    agent_folders = [os.path.join(current_folder,str(i)) for i in range(0,nb_ghosts+1)]
    agent_counters = np.empty(nb_ghosts+1)
    for i in range(nb_ghosts+1):
      try:
        agent_counters[i] = len(os.listdir(agent_folders[i]))
      except FileNotFoundError:
        agent_counters[i] = 0
      for i,lim in enumerate(agent_counters):
        sys.stdout.write("{}\n".format("pacman" if not i else "ghost{}".format(i)))
        sys.stdout.flush()
        one_steps = []
        for count in range(int(lim)):
          sys.stdout.write("\r{}/{}      ".format(count+1,lim))
          sys.stdout.flush()
          try:
            with open(os.path.join(agent_folders[i],str(count)+'.save'),'rb') as f:
              ls = load(f)

              for j,onestep in enumerate(ls):
                one_steps.append(onestep)
              if len(one_steps) >= 30000:
                x,y = computeFittedQIteration(one_steps,N=60,
                                              mlAlgo=ExtraTreesRegressor(n_estimators=10,n_jobs=nb_cores))
                one_steps = []
                master_networks[i].n_estimators += 10
                master_networks[i].fit(x,y)
          except FileNotFoundError:
            pass
        if len(one_steps):
          x,y = computeFittedQIteration(one_steps,N=60,
                                        mlAlgo=ExtraTreesRegressor(n_estimators=100,n_jobs=nb_cores))
          master_networks[i].n_estimators += 10
          master_networks[i].fit(x,y)

        for agents in parallel_agents:
            agents[i].learning_algo = deepcopy(master_networks[i])
        print()
    nb_it = 0
    consec_wins = 0

    while nb_it<100 or abs(consec_wins)<50:

        for i in range(nb_ghosts+1):
            print("Pacman" if not i else "Ghost {}".format(i))

            curr_round = rounds if i else max(rounds,factor_pacman*rounds*nb_ghosts)
            win = False
            nb_try = 0
            while not win:

                for agents in parallel_agents:
                    agents[i].startLearning()

                for j in range(curr_round):
                    sys.stdout.write("\r{}/{}       ".format(j+1,curr_round))
                    sys.stdout.flush()

                    args = [{"layout":layout_instance,
                             "pacman":parallel_agents[a][0],
                             "ghosts":parallel_agents[a][1:],
                             "display":display,
                             "numGames":1,
                             "record":False,
                             "numTraining":1,
                             "timeout":30} for a in range(num_parallel)]

                    results = pool.map(runGamesFQI,args)
                    scores = []
                    for a in range(len(parallel_agents)):
                        scores.append(results[a][0][0].state.data.score)
                        parallel_agents[a] = [results[a][1]["pacman"]]+results[a][1]["ghosts"]

                    gc.collect()

                one_step_transistions = []
                for agents in parallel_agents:
                    one_step_transistions.extend(agents[i].get_History())
                    agents[i].stopLearning()

                sys.stdout.write("\nFQI \n")
                sys.stdout.flush()

                x,y = computeFittedQIteration(one_step_transistions,N=60,
                                                   mlAlgo=ExtraTreesRegressor(n_estimators=10,n_jobs=nb_cores))
                master_networks[i].n_estimators += 10
                master_networks[i].fit(x,y)
                print(len(master_networks[i].estimators_))
                for agents in parallel_agents:
                    agents[i].learning_algo = deepcopy(master_networks[i])

                sys.stdout.write("Final result\n")
                sys.stdout.flush()
                parallel_agents[0][i].showLearn()
                if display_mode != 'quiet' and display_mode != 'text':
                  games = pacman.runGames(layout_instance,parallel_agents[0][0],parallel_agents[0][1:],display,1,False,timeout=30)
                else:
                  os.makedirs(folder,exist_ok=True)
                  games = pacman.runGames(layout_instance,parallel_agents[0][0],parallel_agents[0][1:],display,1,True,timeout=30,
                                          fname=folder+'/agent_{}_nbrounds_{}_{}.pickle'.format(i,nb_it,nb_try))
                parallel_agents[0][i].showLearn(False)
                # Compute how many consecutive wins ghosts or pacman have
                # consec_wins is negative if the ghosts have won, positive otherwise.
                # abs(consec_wins) is the number of consecutive wins.
                for game in games:
                    if game.state.isWin():
                        if not i:
                          win = True
                        consec_wins = max(1,consec_wins+1)
                    else:
                        if i:
                          win = True
                        consec_wins = min(-1,consec_wins-1)
                if not win and not parallel_agents[0][i].round_training:
                    for agents in parallel_agents:
                        agents[i].round_training = 0
                elif parallel_agents[0][i].round_training:
                    print("round_training {}".format(parallel_agents[0][i].round_training))
                    win = False

                if display_mode != 'quiet' and display_mode != 'text':
                    makeGif(folder,'agent_{}_nbrounds_{}_{}.mp4'.format(i,nb_it,nb_try))
                    graphicsDisplay.FRAME_NUMBER = 0
                nb_try += 1
        nb_it += 1

    return master_networks


def readAndDelete(imageName):
    image = io.imread(imageName)
    os.remove(imageName)
    return image

def makeGif(folder='videos',filename='movie.mp4'):
    filename = folder+'/'+filename
    # The images to use are in subfolder frames of the current folder:
    nb_frames = len(os.listdir(os.getcwd()+"/frames"))
    pool = ThreadPool()

    os.makedirs(folder,exist_ok=True)

    with io.get_writer(filename, mode='I',macro_block_size=None) as writer:
        filenames = ['frames/frame_%08d.ps' % i for i in range(nb_frames)]
        images = pool.map(readAndDelete,filenames)
        for image in images:
            writer.append_data(image)


if __name__ == "__main__":

    print("FQI")
    master_nwk = iterativeA3cFQI(nb_ghosts=1,round_training=0,
                                 rounds=50,display_mode='graphics'
                                 ,num_parallel=12,nb_cores=12,folder='FQI')
