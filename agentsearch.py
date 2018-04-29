from pacman import Directions
from game import Agent
import util
from math import inf
# XXX: You should complete this class for Step 1


class Agentsearch(Agent):
    def __init__(self, index=0, time_eater=40, g_pattern=-1, ghost_call = False):
        """
        Arguments:
        ----------
        - `index`: index of your agent. Leave it to 0, it has been put
                   only for game engine compliancy
        - `time_eater`: Amount of time pac man remains in `eater`
                        state when eating a big food dot
        - `g_pattern`: Ghosts' pattern in-game. See agentghost.py.
                       Not useful in this class, value does not matter
        """
        self.actions=[]
        self.index=-1
        self.dead_ends={}   # self.dead_ends[i][0] : dead ends with food
                            # self.dead_ends[i][1] : supposed no dead ends
        self.enable_subopt = True
        self.ghost_call = ghost_call
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

        if self.ghost_call:
            self.enable_subopt = True
            self.actions = []
            self.index = -1
        elif state.getNumFood()>25:
            self.enable_subopt = True
        else:
            self.enable_subopt = False
        
        self.index=self.index+1
    
        if self.index == len(self.actions):
            self.index = 0
            self.actions = self.a_star(state)

        
        return self.actions[self.index]
    
    
    def heuristic1(self,state):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        Return:
        -------
        - The number of foods in state 'state'
        """
        return state.getNumFood()
    
        
    def heuristic2(self,pos_pac,foods): 
        """
        Parameters:
        -----------
        - `pos_pac`: a position
        - `foods`: a list of positions

        Return:
        -------
        - The smallest manhattan distance between 'pos_pac' 
          and the positions in 'foods'
        """
        smallest_dist = inf
        for food in foods:
            new_dist = util.manhattanDistance(pos_pac, food)
            if new_dist < smallest_dist :
                smallest_dist = new_dist
        return smallest_dist
    
    
    def a_star(self, state):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        Return:
        -------
        - The best set of actions according to astar algorithm, using 
          heuristic2 as heuristic if self.enable_subopt is true, 
          using heuristic1 otherwise.
        """
        frontier = util.PriorityQueue()
        frontier.push((state,Directions.STOP), 0 + self.heuristic1(state) )
        # visited_nodes[node] is True if the node has been visited, 
        # a pair(actions,actions_score) if the parent of the node 
        # has been visited, None otherwise:
        visited_nodes = {}
        # Passing again at the same position without having eaten any dot makes no sense:
        key = (state.getFood(), state.getPacmanPosition())
        visited_nodes[key] = ([],0,0)

        while not frontier.isEmpty():
            
            # Get the state with the lower score:
            (curr_state, prec_action) = frontier.pop()
            
            key = (curr_state.getFood(), curr_state.getPacmanPosition())
            
            # If the state has alrady been visited we go to the next state:
            if key in visited_nodes and visited_nodes[key]==True:
                continue
            
            # If we are in the suboptimal mode and if we have eaten food:
            if self.enable_subopt and visited_nodes[key][2]>0:
                    return visited_nodes[key][0]
            # If what we popped is the goal, we can return the sequence of moves:
            if curr_state.getNumFood() == 0:
                return visited_nodes[key][0]
   
            
            (prec_actions, prec_actions_cost, prec_num_eat_food) = visited_nodes[key]
            actions = curr_state.getLegalPacmanActions()
            
            # If we can only go back or continue in the path:
            if len(actions) == 3:
                not_reverse = []
                for action in actions:
                    if action == Directions.REVERSE[prec_action] or action == Directions.STOP:
                        continue
                    else:
                        not_reverse.append(action)
                    
                # If we have food when not going back, we do not want to go back
                # and so do not even consider it:
                if self.hasFood(curr_state, not_reverse):
                    actions = not_reverse
        
            # If we have an intersection, select the dead end if possible:
            elif len(actions) > 3:  
                actions =  self.find_dead_ends(curr_state, actions, prec_action)

            for action in actions:
                if action == Directions.STOP : 
                    continue
                
                # If pacman wants to go in the reverse direction while there is still food just in front of him:
                if action == Directions.REVERSE[prec_action] and self.hasFood(curr_state, prec_action) :
                    continue
                
                child = curr_state.generatePacmanSuccessor(action)
                key_child = (child.getFood(), child.getPacmanPosition())
                # If the state has alrady been visited we go to the next state:
                if key_child in visited_nodes and visited_nodes[key_child]==True:
                    continue
                
                action_cost = self.getCostOfAction(curr_state, action)
                num_eat_food = prec_num_eat_food + int( self.hasFood(curr_state, action) )
                #If suboptimal:
                if self.enable_subopt:
                    foods_pos = child.getFood().asList()
                    score = action_cost + prec_actions_cost + self.heuristic2(child.getPacmanPosition(),foods_pos)
                else:
                    score = action_cost + prec_actions_cost + self.heuristic1(child)

                visited_nodes[key_child] = (prec_actions+
                             [child.getPacmanState().getDirection()], 
                             action_cost+prec_actions_cost, num_eat_food)
                frontier.push( (child,action), score)
                
            # Remember that the state has been visited:
            visited_nodes[key] = True
        
        #Should never happen
        return [state.getLegalPacmanActions()[1]]
        
    def getCostOfAction(self, state, action):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `action`: a pacman action

        Return:
        -------
        - 1 if pacman will eat a food by performing the action 'action', 
          2 otherwise
        """
        if self.hasFood(state, action):
            return 1
        # If there is no food in the place we go:
        return 2
            
    def hasFood(self,state,action):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `action`: a pacman action

        Return:
        -------
        - True if pacman will eat a food by performing the action 'action', 
          False otherwise
        """
        legal = state.getLegalPacmanActions()
        if action in legal: 
            next_state = state.generatePacmanSuccessor(action)
            (x, y) = next_state.getPacmanPosition()
            if state.hasFood(x, y):
                return True
        return False

    
    def find_dead_ends(self,curr_state, actions, prec_action):
        """
        Parameters:
        -----------
        - `curr_state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `actions`: the action that pacman will play next
        - `prec_action`: True if at least one food has been found while the past 
                  exploring of this potential dead end, false otherwise.

        Return:
        -------
        - The set of actions that are worth to consider, by keeping only those 
          that lead to dead ends with food if any, or by deleting those that
          lead to a dead end without food otherwise.
        """
        # If we have already analyzed this intersection:
        pos = curr_state.getPacmanPosition()
        
        # Else, analyze it:        
        if not( pos  in self.dead_ends):
            i=0
            self.dead_ends[pos] =[[],[]]
            for action in actions:
                i=i+1
                if action == Directions.REVERSE[prec_action]:
                    self.dead_ends[pos][1].append(action)
                    continue
                elif action == Directions.STOP:
                    continue
                (first_food, dead_end) = self.is_dead_end(curr_state, action, False)
                # If it is a dead end containing food, go into it:
                if dead_end == (True, True):
                    self.dead_ends[pos][0].append((first_food,action))
                # Else, consider only what s not a dead end:    
                elif not dead_end :
                    self.dead_ends[pos][1].append(action)
                    
        # If we are suboptimal and we have already eaten some food, 
        # return path found until this intersection without dead end containing food:            
        if len(self.dead_ends[pos][0]) == 0:
            actions = self.dead_ends[pos][1]

        else:
            food_dead_ends = self.dead_ends[pos][0]
            actions = []
            for f_d_e in food_dead_ends:
                (x,y) = f_d_e[0]
                if curr_state.hasFood(x,y) :
                    actions = [f_d_e[1]]
                    break
            if actions == []:
                actions = self.dead_ends[pos][1]
                 
        return actions
                
    def is_dead_end(self, state, action_todo, food):
        """
        Parameters:
        -----------
        - `state`: the current game state as defined pacman.GameState.
                   (you are free to use the full GameState interface.)
        - `action_todo`: the action that pacman will play next
        - `food`: True if at least one food has been found while the past 
                  exploring of this potential dead end, false otherwise.

        Return:
        -------
        - (None,False) if 'action_todo' does not lead to a dead end or leads 
                       to a dead end with several branches.
          (pos,(True, fd)) if 'action_todo' leads to a dead end. With fd
          being true if there is a food on the rest of the path or if 'food' 
          is True, and with pos being the position of the food in the remaining
          path if 'food' is false,  None otherwise.
        """
        if self.hasFood(state, action_todo):
            food = True
        next_state = state.generatePacmanSuccessor(action_todo)
        pos = next_state.getPacmanPosition()
        actions = next_state.getLegalPacmanActions()
        if len(actions) == 3:            
            for action in actions:
                    if action == Directions.REVERSE[action_todo] or action == Directions.STOP:
                        continue
                    else:
                        not_reverse = action
                        break
            if food: 
                response = self.is_dead_end(next_state, not_reverse, food)
                return (pos, response[1])
            else:
                return self.is_dead_end(next_state, not_reverse, food)                    
                    
        elif len(actions) == 2:
            # It is a dead end
            if food:
                return (pos,(True, food)) 
            else:
                return (None,(True, food)) 
        else:
            # Either is not a dead end or is a dead end with several branches:
            return (None,False)      