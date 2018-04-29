from pacman import Directions
from game import Agent
from math import inf
import util
import agentsearch
import copy
from math import sqrt
from game import Actions

# XXX: You should complete this class for Step 2


class Agentghost(Agent):
    def __init__(self, index=0, time_eater=40, g_pattern=0):
        """
        Arguments:
        ----------
        - `index`: index of your agent. Leave it to 0, it has been put
                   only for game engine compliancy
        - `time_eater`: Amount of time pac man remains in `eater`
                        state when eating a big food dot
        - `g_pattern`: Ghosts' pattern in-game :
                       0 - leftyghost
                       1 - greedyghost
                       2 - randyghost
                       3 - rpickyghost
        """
        self.g_pattern = g_pattern
        self.max_depth = 0
        self.first_call = True
        self.prob_patt = {}
        self.first_pacman_move = None # The current first pacman move of the expectimax
        # self.prob_prec_move[agent] =
        # (prec_pos_agent, prec_possible_actions, prec_possible_actions_probas)
        # If pattern = 3
        self.prob_prec_move = {}
        self.agentsearch = agentsearch.Agentsearch(ghost_call = True)
        self.min_ghost_dist = 7
        self.prec_moves = [] # Precedent moves of pacman
        self.prec_eat = [] # Precedent foods/capsules eaten by pacman
        # Initially: self.possible_patterns[ghost]=[0,1,2] : 0=lefty, 1=greedy, 2=randy
        self.possible_patterns = {}
        self.listen_search = [False,0]  # True if pacman must listen to the search agent.
        self.prec_pos = [] # Prec positionsof pacman
        pass

    def getAction(self, state):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)

        Return:
        -------
        - A legal move as defined game.Directions.
        """

        num_agents = state.getNumAgents()

        # If there are no ghosts, use the search agent normally:
        if num_agents == 1:
            if self.first_call:
                self.agentsearch = agentsearch134450131558.Agentsearch134450131558(ghost_call = False)
            return self.agentsearch.getAction(state)
        # If ghost far enough, call agentsearch:
        if self.ghostFar(state):
            return self.agentsearch.getAction(state)

        # If the ghost is rpicky:
        if self.g_pattern == 3 and not self.first_call:


            last_pacman_move = self.prec_moves[len(self.prec_moves)-1]

            for ghost in range(1,num_agents):

                # If we know already the patern, no need to go there:
                if len(self.possible_patterns[ghost]) == 1:
                    continue
                move = state.getGhostState(ghost).getDirection()

                # If the ghost has just ben killed, nothing to deduce:
                if move == Directions.STOP:
                    continue
                # If the ghost has been skipped because it was
                # too far from pacman, nothing to deduce:
                if not(ghost in self.prob_prec_move[last_pacman_move]):
                    continue

                updated_possible_patterns = []


                self.prob_prec_move[last_pacman_move]
                self.prob_prec_move[last_pacman_move][ghost][2]
                index_move = self.prob_prec_move[last_pacman_move][ghost][2].index(move)

                for pattern in self.possible_patterns[ghost]:

                    # Update the probabilities using the new actions
                    self.prob_patt[ghost][pattern] = self.prob_patt[ghost][pattern] * self.prob_prec_move[last_pacman_move][ghost][3][pattern][index_move]
                    # If the probability of doing the move that has been done
                    # was hardly zero for this pattern, this is very likely not
                    # the good pattern, else we must still consider it:
                    if self.prob_prec_move[last_pacman_move][ghost][3][pattern][index_move] > 0.0001:
                        updated_possible_patterns.append(pattern)
                    else:
                        self.prob_patt[ghost][pattern] = 0

                self.possible_patterns[ghost] = updated_possible_patterns


                sum_new_probas = sum(self.prob_patt[ghost])
                #Normalize the updated probabilities
                self.prob_patt[ghost] = [i/sum_new_probas for i in self.prob_patt[ghost]]

            update = True
            for ghost in range(1,num_agents):
                # If the pattern is not determined yet or if the pattern is randy:
                if len(self.possible_patterns[ghost]) != 1 or (len(self.possible_patterns[ghost]) == 1 and self.possible_patterns[ghost][0] == 2 ):
                    update = False
                    break
            # If all the patterns are determined and none of them is randy, we can pick a bigger max_depth:
            if update:
                ref_number = 25
                self.max_depth = ref_number - ref_number%state.getNumAgents()

        # If it is the first time that getAction is called, initialize variables:
        if self.first_call:

            # If the ghost is random
            if self.g_pattern == 3:
                self.prob_patt = {}
                for ghost in range(1,num_agents):
                    self.prob_patt[ghost] = [1/3,1/3,1/3]
                    self.possible_patterns[ghost] = [0,1,2]
            if self.g_pattern < 2:
                ref_number = 25
            else:
                ref_number = 15
            self.max_depth = ref_number - ref_number%state.getNumAgents()
            self.first_call = False

        # If the ghost is rpicky:
        if self.g_pattern == 3:
            for direct in ['North','South','West','East']:
                for ghost in range(1,num_agents):
                    self.prob_prec_move[direct] = {}


        result = self.expectimax(state, 0, self.max_depth, [] )[1]
        pac_pos = state.getPacmanPosition()

        # Search only here to let the guess happen:
        if self.listen_search[0]:
            self.listen_search[1] = self.listen_search[1] - 1
            if self.listen_search[1] == 0:
                self.listen_search[0] = False
            result = self.agentsearch.getAction(state)

        # Handle the go and go back issue:
        elif len(self.prec_pos) > 4:
            # if we went a step in a direction
            # and now we want to go in the opposite without any reason, listen to
            # the search agent for the two next steps:
            if result == self.prec_pos[len(self.prec_pos)-2] and not self.prec_eat[len(self.prec_pos)-1] and self.ghostFar(state):
                self.listen_search = [True,3]
                result = self.agentsearch.getAction(state)
            # If there are ghosts near of pacman I am a bit more careful:
            elif (result == self.prec_pos[len(self.prec_pos)-2] == self.prec_pos[len(self.prec_pos)-4] and
                                           self.prec_pos[len(self.prec_pos)-1] == self.prec_pos[len(self.prec_pos)-3] == self.prec_pos[len(self.prec_pos)-5]):
                self.listen_search = [True,3]
                result = self.agentsearch.getAction(state)

        #child = state.generatePacmanSuccessor(result)
        direction = Actions._directions[result]
        next_pac_pos = (pac_pos[0] + direction[0], pac_pos[1] + direction[1])
        pos_capsules = state.getCapsules()
        if state.hasFood(next_pac_pos[0], next_pac_pos[1]) or (next_pac_pos in pos_capsules):
            self.prec_eat.append(True)
        else:
            self.prec_eat.append(False)

        self.prec_pos.append(pac_pos)
        self.prec_moves.append(result)

        return result

    def expectimax(self, state, agent, max_depth, to_skip):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of the agent that should play next
        - `max_depth`: The number of agents'moves that can be performed from
                       the state 'state' before we stop performing moves.
        - `to_skip`: a list of agent's id whose moves have no importance
                     for pacman for the rest of this branch.

        Return:
        -------
        - a pair whose first element is the heuristic value of 'state'
          and whose second is the best action pacman can do according
          to expectimax algorithm if pacman is the next agent to play,
          None if the next agent to play is a ghost.
        """
        agent = agent % state.getNumAgents()

        if max_depth == 0 or state.isWin() or state.isLose():
            return (self.evaluation(state),None)

        #If pacman plays:
        if agent == 0:
            return self.pacman(state, agent, max_depth,copy.deepcopy(to_skip))
        #If a ghost plays:
        else:
            # If the ghost is too far from pacman in this branch of the tree, skip it:
            if agent in to_skip:
                return self.expectimax(state, agent+1, max_depth-1,
                                       copy.deepcopy(to_skip))

            ghost_pos = state.getGhostPosition(agent)
            pos_pacman = state.getPacmanPosition()
            dist = util.manhattanDistance(ghost_pos, pos_pacman)

            # If the ghost is too far from pacman, we will not consider
            # it in any of the child states:
            if dist > self.min_ghost_dist:
                new_to_skip = copy.deepcopy(to_skip)
                new_to_skip.append(agent)
                return self.expectimax(state, agent+1, max_depth-1,
                                       new_to_skip)
            return self.ghost(state, agent, max_depth, copy.deepcopy(to_skip))


    def pacman(self, state, agent, max_depth, to_skip):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of pacman (0), which should play next.
        - `max_depth`: The number of agents'moves that can be performed from
                       the state 'state' before we stop performing moves.
        - `to_skip`: a list of agent's id whose moves have no importance
                     for pacman for the rest of this branch.

        Return:
        -------
        - a pair whose first element is the heuristic value of 'state'
          and whose second is the best action pacman can do according
          to expectimax algorithm.
        """
        actions = state.getLegalPacmanActions()
        best_value = -inf
        best_action = None
        for action in actions:
            if action == Directions.STOP :
                continue
            child = state.generatePacmanSuccessor(action)
            (x,y) = child.getPacmanPosition()
            if max_depth == self.max_depth:
                # Change the curren first pacman action:
                self.first_pacman_move = action
            value = self.expectimax(child, agent+1, max_depth-1, copy.deepcopy(to_skip) )[0]

            if value > best_value:
                best_value = value
                best_action = action
        return (best_value, best_action)


    def ghost(self, state, agent, max_depth, to_skip):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of the agent that should play next, which should not
                   be the one of pacman.
        - `max_depth`: The number of agents'moves that can be performed from
                       the state 'state' before we stop performing moves.
        - `to_skip`: a list of agent's id whose moves have no importance
                     for pacman for the rest of this branch.

        Return:
        -------
        - a pair whose first element is the heuristic value of 'state'
          and whose second is None.
        """
        actions = state.getLegalActions(agent)
        probabilities = self.probabilities(state, agent, actions, max_depth)
        i = 0
        mean_value = 0
        for action in actions:

            if probabilities[i] == 0:
                i+=1
                continue
            child = state.generateSuccessor(agent, action)
            value = self.expectimax(child, agent+1, max_depth-1, copy.deepcopy(to_skip) )[0]
            mean_value += value * probabilities[i]
            i+=1
        return (mean_value, None)


    def evaluation(self, state):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        Return:
        -------
        - a heuristic value evaluating how good the current state is.
        """
        pos = state.getPacmanPosition()
        game_score = state.getScore()

        if state.isLose() or state.isWin():
            return state.getScore()

        # food distance
        food_list = state.getFood().asList()
        num_food = len(food_list)
        dist_closest_food = min(map(lambda x: util.manhattanDistance(pos, x), food_list))

        numberOfCapsulesLeft = len(state.getCapsules())

        # active ghosts are ghosts that aren't scared.
        scared_ghosts = []
        not_scared_ghosts = []
        for ghost in state.getGhostStates():
            if not ghost.scaredTimer:
                not_scared_ghosts.append(ghost)
            else:
                scared_ghosts.append(ghost)

        distances_ghost = dist_closest_scared_ghost = distances_active_ghosts = 0

        if not_scared_ghosts:
            distances_ghost = [ util.manhattanDistance(pos, curr_ghost.getPosition()) for curr_ghost in not_scared_ghosts]
            distances_active_ghosts = sum([ 1./dist for dist in distances_ghost ])

        if scared_ghosts:
            dist_closest_scared_ghost = min(map(lambda g: util.manhattanDistance(pos, g.getPosition()), scared_ghosts))
        else:
            dist_closest_scared_ghost = 0


        score = -0.5 * dist_closest_food + \
                  10    * distances_active_ghosts + \
                  -20    * dist_closest_scared_ghost + \
                  -10 * numberOfCapsulesLeft + \
                  +20    * (1./sqrt(num_food)) +\
                  +1    * game_score

        return score


    def probabilities(self, state, agent, actions, max_depth):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of the agent that should play next
        - `actions`: the possible actions of the agent 'agent'
        - `max_depth`: The number of agents'moves that can be performed from
                       this state before we stop performing moves.

        Return:
        -------
        - a list of probabilities 'probas' such that probas[i] is the
          probability that the ith action of 'actions' will be played by
          the agent when it will play next according to our beliefs about
          the agent (pattern, and past evidences if pattern is rpicky).
          Updates self.prob_prec_move if we are in the first round
          (ie if an agent has not yet played twice during the
          current expectimax)
        """


        # Get the precedent direction of the agent to handle
        # the go-back interdiction
        ghost = state.getGhostState(agent)
        prec_action = ghost.getDirection()

        # If only one action possible, we can do it, even if it is turning around:
        if len(actions) == 1:
            probas = [1]
            # If rpickyghosts we must still update dictionnaries:
            if self.g_pattern == 3:
                if self.max_depth-max_depth < state.getNumAgents():
                    prob_lefty = prob_greedy = prob_randy = probas
                    self.prob_prec_move[self.first_pacman_move][agent] = (state.getGhostPosition(agent),
                                       agent,
                                       actions,
                                       (prob_lefty, prob_greedy, prob_randy))

        # If leftyghost:
        elif self.g_pattern == 0:
            probas = self.probaLefty(state, agent, actions, prec_action )

        # If greedyghost:
        elif self.g_pattern == 1:
            probas = self.probaGreedy(state, agent, actions, prec_action )

        # If randyghost:
        elif self.g_pattern == 2:
            probas = self.probaRandy(state, agent, actions, prec_action )

        # If unknown pattern:
        else:

            prob_lefty = self.probaLefty(state, agent, actions, prec_action )
            prob_greedy = self.probaGreedy(state, agent, actions, prec_action )
            prob_randy = self.probaRandy(state, agent, actions, prec_action )
            prob = self.prob_patt[agent]

            # Keep the posibles actions and their probabilites if we are at
            # the first turn in order to be able to figure out
            # the followed pattern:
            if self.max_depth-max_depth < state.getNumAgents():
                self.prob_prec_move[self.first_pacman_move][agent] = (state.getGhostPosition(agent),
                                   agent,
                                   actions,
                                   (prob_lefty, prob_greedy, prob_randy))
            probas = [ prob[0]*l + prob[1]*g
                      + prob[2]*r for l,g,r
                      in zip(prob_lefty, prob_greedy, prob_randy)]

        return probas


    def probaLefty(self, state, agent, actions, prec_action ):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of the agent that should play next
        - `actions`: the possible actions of the agent 'agent'
        - `prec_action`: the action that was played by the agent 'agent'
                         in the precedent move.

        Return:
        -------
        - a list of probabilities 'probas' such that probas[i] is the
          probability that the ith action of 'actions' will be played by
          the agent when it will play next if the agent follows the lefty
          pattern.
        """
        probas = [0] * len(actions)
        direction = Directions.WEST

        while True:
            if direction in actions and direction != Directions.REVERSE[prec_action]:
                probas[actions.index(direction)] = 1
                return probas
            else:
                direction = Directions.LEFT[direction]
                if direction == Directions.WEST:
                    direction = Directions.STOP

    def probaGreedy(self, state, agent, actions, prec_action ):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of the agent that should play next
        - `actions`: the possible actions of the agent 'agent'
        - `prec_action`: the action that was played by the agent 'agent'
                         in the precedent move.

        Return:
        -------
        - a list of probabilities 'probas' such that probas[i] is the
          probability that the ith action of 'actions' will be played by
          the agent when it will play next if the agent follows the greedy
          pattern.
        """
        probas = [0] * len(actions)
        pos_pacman = state.getPacmanPosition()
        pos_ghost = state.getGhostPosition(agent)

        list_dist = [0] * len(actions)
        ghost_state = state.getGhostState(agent)
        is_scared = ghost_state.scaredTimer > 0

        if is_scared:
            comparator = lambda x,y: x > y
            best_dist = -inf
            speed = 0.5
        else:
            comparator = lambda x,y: x < y
            best_dist = inf
            speed = 1
        i=0

        for action in actions:

            direction = Actions._directions[action]
            displacement = tuple(x*speed for x in direction)
            next_pos_ghost = (pos_ghost[0] + displacement[0], pos_ghost[1] + displacement[1])


            dist = util.manhattanDistance(next_pos_ghost, pos_pacman)
            list_dist[i] = dist
            i+=1
            if comparator(dist,best_dist) and action != Directions.REVERSE[prec_action]:
                best_dist = dist

        best_actions = [ action for action,
                        distance in zip(actions, list_dist)
                        if distance == best_dist]
        nb_best = len(best_actions)
        for action in best_actions:
            probas[actions.index(action)] = 1/nb_best
        return probas

    def probaRandy(self, state, agent, actions, prec_action ):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `agent`: the id of the agent that should play next
        - `actions`: the possible actions of the agent 'agent'
        - `prec_action`: the action that was played by the agent 'agent'
                         in the precedent move.

        Return:
        -------
        - a list of probabilities 'probas' such that probas[i] is the
          probability that the ith action of 'actions' will be played by
          the agent when it will play next if the agent follows the randy
          pattern.
        """
        nb_actions = len(actions)
        probas = [0] * nb_actions

        for action in actions:
            probas[actions.index(action)] = 1/nb_actions

        prob_lefty = self.probaLefty(state, agent, actions, prec_action )
        prob_greedy = self.probaGreedy(state, agent, actions, prec_action )

        probas = [ 0.25*l + 0.5*g + 0.25*r for l,g,r
                  in zip(prob_lefty, prob_greedy, probas)]
        return probas

    def ghostFar(self,state):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)

        Return:
        -------
        - True if all the ghosts are further from pacman than
          self.min_ghost_dist, False otherwise.
        """
        pos_pacman = state.getPacmanPosition()
        num_agents = state.getNumAgents()
        for ghost in range(1,num_agents):
            ghost_pos = state.getGhostPosition(ghost)
            dist = util.manhattanDistance(ghost_pos, pos_pacman)
            if dist < self.min_ghost_dist:
                return False
        return True