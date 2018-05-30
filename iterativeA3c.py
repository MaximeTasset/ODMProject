
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:38:41 2018

@author: Sarah and Maxime
"""
import pacman
from reinfagent import ReinfAgent,ReinfAgentFQI,getDataState,computeFittedQIteration
import layout

import tensorflow as tf
from brain import AC_Network
import numpy as np
import sys
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool,Process,Queue
import psutil
import os
import imageio as io
#from tensorflow.python.client import device_lib
import gc

#from sklearn.base import clone
from copy import deepcopy
from sklearn.ensemble import ExtraTreesRegressor

from pickle import load
#GPU = False
#for d in device_lib.list_local_devices():
#    if d.device_type == 'GPU':
#        GPU = True
#        break
#if GPU:
#    # Assume that you have 8GB of GPU memory and want to allocate ~6GB:
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
#    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def runGames(kargs):
    return pacman.runGames(**kargs)
#def runGames2(kargs,queue):
#    queue.put(pacman.runGames(**kargs))
#
#def testfunc(q):
#  print(q)
#  return


def iterativeA3c(nb_ghosts=3,display_mode='graphics',
                 round_training=5,rounds=100,num_parallel=1,nb_cores=-1, folder='videos',layer='mediumClassic',vector=True,loadFrom='games'):

    tf.reset_default_graph()
    pool = ThreadPool(nb_cores)
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


    layout_instance = layout.getLayout(layer)
    nb_ghosts = min(len(layout_instance.agentPositions)-1,nb_ghosts)

    # Get the length of a state:
    init_state = pacman.GameState()
    init_state.initialize(layout_instance, nb_ghosts)
    s_size = len(getDataState(init_state,vector=vector))
    if vector:
        grid_size = init_state.getWalls().width,init_state.getWalls().height,5
    else:
        grid_size = init_state.getWalls().width,init_state.getWalls().height
    # Generate global network
    master_networks = [AC_Network(s_size,4 if i else 5,grid_size,"global_"+str(i),None,
                                  global_scope="global_"+str(i))
                                    for i in range(0,nb_ghosts+1)]
    global_episodes = [tf.Variable(0,dtype=tf.int32,name='global_episodes'+str(i),trainable=False) for i in range(0,nb_ghosts+1)]
    optims = [tf.train.AdamOptimizer(learning_rate=1e-5) for i in range(0,nb_ghosts+1)]


    with tf.Session() as sess:

        parallel_agents = [[ReinfAgent(optims[i],global_episodes[i],sess,
                                       s_size,4 if i else 5,grid_size,index=i,
                                       name="worker_{}_{}".format(i,j),
                                       global_scope="global_"+str(i),
                                       round_training=round_training if i else max(round_training,round_training*nb_ghosts),
                                       vector=vector)
                                        for i in range(0,nb_ghosts+1)]
                                        for j in range(num_parallel)]

        sess.run(tf.global_variables_initializer())
        main_agents = parallel_agents[0]
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
            for count in range(int(lim)):
              sys.stdout.write("\r{}/{}       ".format(count+1,lim))
              sys.stdout.flush()
              try:
                with open(os.path.join(agent_folders[i],str(count)+'.save'),'rb') as f:
                    ls = load(f)
                    for j,onestep in enumerate(ls):
                        main_agents[i].add_transition(onestep[:-1],not len(onestep[-1]))
              except FileNotFoundError:
                pass
            print()
            break
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
#        queue = Queue()
        for i in range(nb_ghosts+1):
          main_agents[i].showLearn()
        if display_mode != 'quiet' and display_mode != 'text':
          games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,False,timeout=30)
        else:
          os.makedirs(folder,exist_ok=True)
          games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,True,timeout=30,
                                              fname=folder+'/initial.pickle')
        for i in range(nb_ghosts+1):
          main_agents[i].showLearn(False)

        if display_mode != 'quiet' and display_mode != 'text':
          makeGif(folder,'initial.mp4')
          graphicsDisplay.FRAME_NUMBER = 0

        while nb_it<100 or abs(consec_wins)<50:

            for i in range(nb_ghosts+1):
                print("Pacman" if not i else "Ghost {}".format(i))

                curr_round = rounds if i else max(rounds,2*rounds*nb_ghosts)
#                with open('save_scores.txt','a') as f:
#                    f.write('agent '+str(i)+'\n')

                win = False
                nb_try = 0
                while not win:
                    for agents in parallel_agents:
                        agents[i].startLearning()

                    for j in range(curr_round):
                        sys.stdout.write("\r                {}/{}       ".format(j+1,curr_round))
                        sys.stdout.flush()


#                        results = []
#                        pr = []
#                        for arg in args:
#                          p = Process(target=runGames2, args=(arg,queue))
#                          p.start()
#                          pr.append(p)
#                          if len(pr) == nb_cores:
#                            pr[0].join()
#                            pr = pr[1:]
#
#                        while not queue.empty():
#                          results.append(queue.get())
#
#                        score = sum([game[0].state.data.score for game in results if len(game)!=0])
                        score = sum([game[0].state.data.score for game in pool.map(runGames,args)
                            if len(game)!=0])

#                        with open('save_scores.txt','a') as f:
#                            f.write(str(score)+'\n')
                        gc.collect()
                    for agents in parallel_agents:
                        agents[i].stopLearning()


                    sys.stdout.write("           Final result       \n")
                    sys.stdout.flush()
                    main_agents[i].showLearn()
                    if display_mode != 'quiet' and display_mode != 'text':
                      games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,False,timeout=30)
                    else:
                      os.makedirs(folder,exist_ok=True)
                      games = pacman.runGames(layout_instance,main_agents[0],main_agents[1:],display,1,True,timeout=30,
                                              fname=folder+'/agent_{}_nbrounds_{}_{}.pickle'.format(i,nb_it,nb_try))
                    main_agents[i].showLearn(False)
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
                    if not win and not main_agents[i].round_training:
                        for agents in parallel_agents:
                            agents[i].round_training = curr_round/2
                    elif main_agents[i].round_training:
                        print("round_training {}".format(main_agents[i].round_training))
                        win = False

                    if display_mode != 'quiet' and display_mode != 'text':
                        makeGif(folder,'agent_{}_nbrounds_{}_{}.mp4'.format(i,nb_it,nb_try))
                        graphicsDisplay.FRAME_NUMBER = 0
                    nb_try += 1
            nb_it += 1


    return master_networks

def runGamesFQI(kargs):
    games = pacman.runGames(**kargs)
    return (games,kargs)

def iterativeA3cFQI(nb_ghosts=3,display_mode='graphics',
                 round_training=5,rounds=100,num_parallel=1,nb_cores=-1, folder='videos',loadFrom='games',layer='mediumClassic'):

    pool = Pool(nb_cores)
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


    layout_instance = layout.getLayout(layer)
    nb_ghosts = min(len(layout_instance.agentPositions)-1,nb_ghosts)


    # Generate global network
    master_networks = [ExtraTreesRegressor(n_estimators=10,n_jobs=nb_cores,warm_start=True) for i in range(0,nb_ghosts+1)]

    factor_pacman = 2

    parallel_agents = [[ReinfAgentFQI(index=i,
                                      round_training=round_training if i else max(round_training,factor_pacman*round_training*nb_ghosts))
                                    for i in range(0,nb_ghosts+1)]
                                    for j in range(num_parallel)]

#    main_agents = parallel_agents[0]
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
          sys.stdout.write("\r{}/{}       ".format(count+1,lim))
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

#    sys.stdout.write("test\n")
#    sys.stdout.flush()
#    for key in args[0]:
#        sys.stdout.write("{}\n".format(args[0][key]))
#        pool.map(test,[arg[key] for arg in args])
#    sys.stdout.write("fin test\n")
#    sys.stdout.flush()
#    sys.exit()

    while nb_it<100 or abs(consec_wins)<50:

        for i in range(nb_ghosts+1):
            print("Pacman" if not i else "Ghost {}".format(i))

            curr_round = rounds if i else max(rounds,factor_pacman*rounds*nb_ghosts)
#            with open('save_scores.txt','a') as f:
#                f.write('agent '+str(i)+'\n')

            win = False
            nb_try = 0
            while not win:

                for agents in parallel_agents:
                    agents[i].startLearning()

                for j in range(curr_round):
                    sys.stdout.write("\r                {}/{}       ".format(j+1,curr_round))
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

#                    with open('save_scores.txt','a') as f:
#                        f.write(str(scores)+ '\t mean: '+ str(sum(scores)/num_parallel)+'\n')
                    gc.collect()

                one_step_transistions = []
                for agents in parallel_agents:
                    one_step_transistions.extend(agents[i].get_History())
                    agents[i].stopLearning()


                sys.stdout.write("\nFQI \n")
                sys.stdout.flush()

#                one_step_transistions = []
#                while not global_episodes[i].empty():
#                    one_step_transistions.append(global_episodes[i].get())

                x,y = computeFittedQIteration(one_step_transistions,N=60,
                                                   mlAlgo=ExtraTreesRegressor(n_estimators=10,n_jobs=nb_cores))
                master_networks[i].n_estimators += 10
                master_networks[i].fit(x,y)
                print(len(master_networks[i].estimators_))
                for agents in parallel_agents:
                    agents[i].learning_algo = deepcopy(master_networks[i])

                sys.stdout.write("           Final result       \n")
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
#        for i in range(nb_frames):
#            filename = 'frames/frame_%08d.ps' % i
#            image = io.imread(filename)
#            writer.append_data(image)
#            os.remove(filename)

def test(p):
  p.append("a")


if __name__ == "__main__":
#  l = []
#  p = Process(target=test,args=(l,))
#  p.start()
#  p.join()
#  print(l)
  if (len(sys.argv) == 2 and int(sys.argv[1]) ):
    print("FQI")
    master_nwk = iterativeA3cFQI(nb_ghosts=1,round_training=0,rounds=50,display_mode='graphics',num_parallel=12,
               nb_cores=12,folder='FQI')
  else:
    print("A3C")
    master_nwk = iterativeA3c(nb_ghosts=1,round_training=0,rounds=200,display_mode='graphics',num_parallel=6,
               nb_cores=12,folder='A3Cvectnoghost_pretraining3_only',vector=True)


#  pool = Pool(4)
#  b = [[a] for a in range(20)]
#  pool.map(test,b)
#  sys.stdout.write("fin {}".format(b))
#  sys.stdout.flush()



#  max(1,psutil.cpu_count())
