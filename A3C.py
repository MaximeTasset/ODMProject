"""
Created on Fri Apr  6 17:12:23 2018

@author: Sarah and Maxime
"""
import numpy as np
import tensorflow as tf
import time, threading,sys,os
import util
#
#from keras import backend as K
import tensorflow.contrib.slim as slim
from multiprocessing.pool import ThreadPool
import pacman
import layout
from game import Agent
from ghostAgents import GhostAgent
from reinfagent import getDataState,DIRECTION,MAX_SIZE
from iterativeA3c import makeGif
from pickle import load

#-- constants
THREADS = 8
OPTIMIZERS = 4

GAMMA = 0.999

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.6
EPS_STOP  = .15
EPS_STEPS = 7500000

MIN_BATCH = MAX_SIZE
LEARNING_RATE = 1e-6

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .01 	# entropy coefficient

NUM_STATE = None
NUM_ACTIONS = None
NONE_STATE = None

GRID_SIZE = None


#---------

class Brain:

   def __init__(self,session,index):
      # train_queue contains 5 queues, which contain respectively:
      # s, a, r, s', s' terminal mask:
      self.train_queue = [ [], [], [], [], [] ]
      self.lock_queue = threading.Lock()
      self.session = session
#      K.set_session(self.session)
#      K.manual_variable_initialization(True)

      self.index = index
      self.make_model()
      self.graph = self.make_optimizer_graph()

      with tf.variable_scope(str(self.index)):
        self.default_graph = tf.get_default_graph()

      self.learn = False

   def setLearn(self,learn):
     self.learn = learn

   def make_model(self):
      with tf.variable_scope(str(self.index)):

        self.inputs = tf.placeholder(shape=[None,NUM_STATE],dtype=tf.float32)

        self.imageIn = tf.reshape(self.inputs,shape=[-1,GRID_SIZE[0],GRID_SIZE[1],1 if len(GRID_SIZE) == 2 else GRID_SIZE[2]])
        self.conv1 = slim.conv2d(activation_fn=tf.nn.tanh,
                inputs=self.imageIn,num_outputs=81,
                kernel_size=tuple([9,9]),stride=tuple([1,1]),padding='VALID')
        self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=9,
                kernel_size=tuple([3,3]),stride=tuple([1,1]),padding='VALID')

        hidden = slim.fully_connected(slim.flatten(self.conv2),100,activation_fn=tf.nn.tanh)
        for i in range(20):
            hidden = slim.fully_connected(hidden,100,activation_fn=tf.nn.tanh,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

        self.policy = slim.fully_connected(hidden,NUM_ACTIONS[self.index],
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.truncated_normal_initializer(0.01),
                biases_initializer=None)
        self.value = slim.fully_connected(hidden,1,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(0.01),
                biases_initializer=None)

   def make_optimizer_graph(self):
      with tf.variable_scope(str(self.index)):
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS[self.index]))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        log_prob = tf.log( tf.reduce_sum(self.policy * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - self.value

        loss_policy = - log_prob * tf.stop_gradient(advantage)
        loss_value  = LOSS_V * tf.square(advantage)
        entropy = LOSS_ENTROPY * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10),
                                               axis=1, keepdims=True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)
        return self.inputs, a_t, r_t, minimize

   def optimize(self):
      if len(self.train_queue[0]) < MIN_BATCH:
         time.sleep(0)
         return
      with self.lock_queue:
         #another thread could already have enter in the lock before.
         if len(self.train_queue[0]) < MIN_BATCH:
            return

         s, a, r, s_, s_mask = self.train_queue
         self.train_queue = [ [], [], [], [], [] ]

      s = np.vstack(s)
      a = np.vstack(a)
      r = np.vstack(r)
      s_ = np.vstack(s_)
      s_mask = np.vstack(s_mask)
      with tf.variable_scope(str(self.index)):
          v = self.predict_v(s_)
          r = r + GAMMA_N * v * s_mask

          s_t, a_t, r_t, minimize = self.graph
          self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

   def train_push(self, s, a, r, s_):
      if not self.learn:
         return

      with self.lock_queue:
         self.train_queue[0].append(s)
         self.train_queue[1].append(a)
         self.train_queue[2].append(r)

         if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
         else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)

   def predict(self, s):
      with self.default_graph.as_default():
         p, v = self.session.run([self.policy,self.value],feed_dict={self.inputs:s})
         return p, v

   def predict_p(self, s):
      with self.default_graph.as_default():
         return self.session.run(self.policy,feed_dict={self.inputs:s})

   def predict_v(self, s):
      with self.default_graph.as_default():
         return self.session.run(self.value,feed_dict={self.inputs:s})

#---------

class Agent(GhostAgent,Agent):
   def __init__(self,index, eps_start, eps_end, eps_steps,brain,nb_ob=100,vector=True):
      self.eps_start = eps_start
      self.eps_end   = eps_end
      self.eps_steps = eps_steps
      self.brain = brain
      self.index = index
      self.nb_ob = nb_ob
      self.vector = vector
      if index:
        from greedyghost import Greedyghost
        self.training_ghost = Greedyghost(index)
      else:
        from agentghost  import Agentghost
        self.training_pacman = Agentghost(index=0, time_eater=0, g_pattern=1)

      print(self.index == brain.index)
      self.show = False

      self.batch = []
      self.R = 0.
      self.frames = 0
      self.learn = False

      self.prev = None

   def showLearn(self,show=True):
      self.show = show

   def startLearning(self):
      self.learn = True

   def stopLearning(self):
      self.learn = False

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
       legal = state.getLegalActions(self.index)
       if not self.show and self.nb_ob:
            if self.index:
              dist = self.training_ghost.getDistribution(state)
              dist.setdefault(0)
              dist_p = np.zeros(len(DIRECTION))
              for i,d in enumerate(DIRECTION):
                dist_p[i] = dist[d]
              move = np.random.choice(dist_p,p=dist_p)
              move = DIRECTION[np.argmax(dist_p == move)]
            else:
              move = self.training_pacman.getAction(state)
       else:
           legalActions = list(map(DIRECTION.index,legal))

           if not self.show and np.random.random() < self.getEpsilon():
             move = DIRECTION[legalActions[np.random.randint(0, len(legalActions))]]
           else:
             s = getDataState(state,self.index,vector=self.vector)

             p = self.brain.predict_p(np.array([s]))[0]

             p = [p[i] if not np.isnan(p[i]) else 0 for i in range(len(p)) if i in legalActions]
             if sum(p):
                 p /= sum(p)
                 move = DIRECTION[np.random.choice(legalActions, p=p)]
             else:
                 move = DIRECTION[np.random.choice(legalActions)]

       if not move in legal:
         legalActions = list(map(DIRECTION.index,legal))
         move = DIRECTION[np.random.choice(legalActions)]

       if self.learn:
         self._saveOneStepTransistion(state,move,False)

       return move
   def final(self,state):
        self._saveOneStepTransistion(state,None,True)
        self.nb_ob = max(0,self.nb_ob-1)

   def _saveOneStepTransistion(self,state,move,final):
        state_data = tuple(getDataState(state,self.index,vector=self.vector).tolist())
        if not self.prev is None:

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

            self.frames += 1

            self.train(self.prev[2],self.prev[1],reward,state_data if not final else None)


        if not final:
          move = DIRECTION.index(move)

          self.prev = (state.deepCopy(),move,state_data)
        else:
          self.prev = None

   def getEpsilon(self):
      if(self.frames >= self.eps_steps):
         return self.eps_end
      else:
         return self.eps_start + self.frames * (self.eps_end - self.eps_start) / self.eps_steps	# linearly interpolate


   def train(self, s, a, r, s_):
      def get_sample(batch, n):
         s, a, _, _  = batch[0]
         _, _, _, s_ = batch[n-1]

         return s, a, self.R, s_

      a_cats = np.zeros(NUM_ACTIONS[self.index])	# turn action into one-hot representation
      a_cats[a] = 1

      self.batch.append( (s, a_cats, r, s_) )

      self.R = ( self.R + r * GAMMA_N ) / GAMMA

      if s_ is None:
         while len(self.batch) > 0:
            n = len(self.batch)
            s, a, r, s_ = get_sample(self.batch, n)
            self.brain.train_push(s, a, r, s_)
            self.R = ( self.R - self.batch[0][2] ) / GAMMA
            self.batch.pop(0)

         self.R = 0

      if len(self.batch) >= N_STEP_RETURN:
         s, a, r, s_ = get_sample(self.batch, N_STEP_RETURN)
         self.brain.train_push(s, a, r, s_)

         self.R = self.R - self.batch[0][2]
         self.batch.pop(0)


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self,brains):
        threading.Thread.__init__(self)
        self.brains = brains

    def run(self):
        while not self.stop_signal:
            for brain in self.brains:
                brain.optimize()

    def stop(self):
        self.stop_signal = True

def runGames(kargs):
    return pacman.runGames(**kargs)



def iterativeA3c(nb_ghosts=3,display_mode='graphics',
                 round_training=5,rounds=100,num_parallel=1,nb_cores=-1, folder='videos',layer='mediumClassic',vector=True,loadFrom='games'):
    global NUM_STATE,NUM_ACTIONS,NONE_STATE,GRID_SIZE
    tf.reset_default_graph()
    pool = ThreadPool(nb_cores)

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
    NONE_STATE = np.zeros_like(getDataState(init_state,vector=vector))
    s_size = len(NONE_STATE)
    NUM_STATE = s_size
    NUM_ACTIONS = [4 if i else 5 for i in range(0,nb_ghosts+1)]

    if vector:
        GRID_SIZE = init_state.getWalls().width,init_state.getWalls().height,5
    else:
        GRID_SIZE = init_state.getWalls().width,init_state.getWalls().height

    with tf.Session() as sess:

        master_networks = [Brain(sess,i) for i in range(0,nb_ghosts+1)]

        sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

        opts = [Optimizer(master_networks) for k in range(OPTIMIZERS)]

        for op in opts:
          op.start()
        try:
            parallel_agents = [[Agent(index=i, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS,
                                      brain=master_networks[i],nb_ob=round_training)
                                            for i in range(0,nb_ghosts+1)]
                                            for j in range(num_parallel)]

            main_agents = parallel_agents[0]

            current_folder = os.path.join(loadFrom,str(nb_ghosts))
            agent_folders = [os.path.join(current_folder,str(i)) for i in range(0,nb_ghosts+1)]
            agent_counters = np.empty(nb_ghosts+1)
            for i in range(nb_ghosts+1):
                try:
                    print(i)
                    agent_counters[i] = len(os.listdir(agent_folders[i]))
                except FileNotFoundError:
                    agent_counters[i] = 0
            c = 0
            for i,lim in enumerate(agent_counters):
                sys.stdout.write("{}\n".format("pacman" if not i else "ghost{}".format(i)))
                sys.stdout.flush()
                master_networks[i].setLearn(True)
                for count in range(int(lim)):
                  sys.stdout.write("\r{}/{}       ".format(count+1,lim))
                  sys.stdout.flush()
                  try:
                    with open(os.path.join(agent_folders[i],str(count)+'.save'),'rb') as f:
                        ls = load(f)
                        for j,onestep in enumerate(ls):
                            s,a,r,s_,p = onestep
                            main_agents[i].train(s, a, r, s_ if len(p) else None)
                            c += 1
                            # To avoid the queue overflow (otherwise it may
                            # lead to crash)
                            if c == 10*MIN_BATCH:
                              c = 0
                              time.sleep(5)
                  except FileNotFoundError:
                    pass
                master_networks[i].setLearn(False)

                print()


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

                    win = False
                    nb_try = 0
                    while not win:
                        for agents in parallel_agents:
                            agents[i].startLearning()
                        master_networks[i].setLearn(True)
                        for j in range(curr_round):
                            sys.stdout.write("\r{}/{}       ".format(j+1,curr_round))
                            sys.stdout.flush()

                            games = pool.map(runGames,args)

                        for agents in parallel_agents:
                            agents[i].stopLearning()
                        master_networks[i].setLearn(False)

                        sys.stdout.write("Final result       \n")
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
                        if not win and not main_agents[i].nb_ob:
                            for agents in parallel_agents:
                                agents[i].nb_ob = int(curr_round/2)
                        elif main_agents[i].nb_ob:
                            print("round_training {}".format(main_agents[i].nb_ob))
                            win = False

                        if display_mode != 'quiet' and display_mode != 'text':
                            makeGif(folder,'agent_{}_nbrounds_{}_{}.mp4'.format(i,nb_it,nb_try))
                            graphicsDisplay.FRAME_NUMBER = 0
                        nb_try += 1
                nb_it += 1
        finally:
            for op in opts:
                op.stop()
            for op in opts:
                op.join()


    return master_networks


if __name__ == "__main__":
    mn = iterativeA3c(nb_ghosts=0,display_mode='graphic',
                 round_training=200,rounds=200,num_parallel=4,nb_cores=4, folder='A3Cv2_only',layer='mediumClassic')