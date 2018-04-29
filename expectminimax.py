from game import Directions
from search import dist
from util import manhattanDistance
import numpy as np
from pacman import SCARED_TIME
from game import Actions
class ExpectMiniMax:
    """
    Class for Expect MiniMax
    """
    def __init__(self, graph, start, goal, g_pattern, moves, maxDepth):
        """
        Inits the class
        :param graph: graph of the map
        :param start: current pacman position
        :param goal: goal position for pacman
        :param g_pattern: ghost pattern
        :param moves: moves to get from start to goal
        :param maxDepth: maximum depth of the minimax relative to pacman
        """
        self. nodes_array = graph.nodes_array
        self.nodes_list = graph.nodes_list
        self.nb_food = graph.nb_food
        self.nb_capsules = graph.nb_capsule
        self.start = start
        self.goal = goal
        self.g_pattern = g_pattern
        self.moves = moves
        self.maxDepth = maxDepth

        if self.g_pattern == 0:
            self.pattern_Except = self.ghost_cc_left
        elif self.g_pattern == 1:
            self.pattern_Except = self.ghost_greedy
        elif self.g_pattern == 2:
            self.pattern_Except = self.ghost_randy
        elif self.g_pattern == 3:
            self.pattern_Except = self.ghost_rpicky
#        print(self.goal)

    def expect(self, current, depth, player, g_pos, food=0,ghost=0, goal_depth = -1):
        """
        ExpectMiniMax, There no min because there is no opponent player
        ATM it only handles leftyghosts
        :param current: Current Node
        :param depth: depth until end of recursion
        :param player: Whether or not it's the player playing
        :param g_pos: ghost positions and orientation (x,y,facing,scared_Timer,start,p_prob)
        :param food: How many foods pacman has consumed until now
        :param ghost: How many ghost pacman has eaten until now
        :param
        :return: (alpha score, list of moves in reverse order)
        """

        # terminal test
        g_pos,g = self.collide(current,g_pos)
        ghost += g

        if current == self.goal:
            goal_depth = max(goal_depth, depth)

        if self.terminal_state(current, g_pos) or depth == 0:
            return self.heuristic_val(current, g_pos, goal_depth, food,ghost, depth), []

        # player is max
        if player:
            flag_food = False
            if current.has_food:
                food += 1
                current.has_food = False
                self.nb_food -= 1
                flag_food = True

            flag_cap = False
            if current.has_capsule:
                current.has_capsule = False
                g_pos = [(x,y,facing,(SCARED_TIME,b),start,p_prob) for x,y,facing,(_,b),start,p_prob in g_pos]
                self.nb_capsules -= 1
                flag_cap = True

            alpha, direction = self.expect(current, depth - 1, False, g_pos, food,ghost, goal_depth)
            direction.append(Directions.STOP)

            for neighbor in current.neighbors:
                m = current.getNeighborDirection(neighbor)
                a, moves = self.expect(neighbor, depth - 1, False, g_pos, food,ghost, goal_depth)
                moves.append(m)

                if a >= alpha:
                    alpha = a
                    direction = moves
            #reinit the current.has_food and current.has_capsule values
            if flag_food:
                current.has_food = True
                self.nb_food += 1
            if flag_cap:
                current.has_capsule = True
                self.nb_capsules += 1

        # random event
        else:
            alpha = 0

            listMovesByGhost = [self.pattern_Except(current,i, g_pos) for i in range(len(g_pos))]
            indexes = np.zeros(len(g_pos),dtype=np.int)
            last_g = len(indexes)-1

            while indexes[0] < len(listMovesByGhost[0]):
                g_pos_next = []
                proba = 1
                for i in range(len(listMovesByGhost)):
                    prob, pos = listMovesByGhost[i][int(indexes[i])]
                    proba *= prob
                    g_pos_next.append(pos)
                indexes[last_g] += 1
                for i in reversed(range(1,len(indexes))):
                    if indexes[i] == len(listMovesByGhost[i]):
                        indexes[i] = 0
                        indexes[i-1] += 1
                    else:
                        break
                a, direction = self.expect(current, depth, True, g_pos_next, food,ghost, goal_depth)
                alpha += proba * a

        return alpha, direction

    def collide(self,current,g_pos):
        """
        Checks for collisions between pacman and the ghosts
        :param current: current node
        :param g_pos: ghost positional information
        :return: next ghost positions and number of collisions
        """
        x = current.i
        y = current.j
        next_g_pos = []
        collide = 0
        for x1,y1,pos,(scared,b),(s1,s2),p_prob in g_pos:
            if x == x1 and y == y1 and scared > 0:
                collide += 1
                next_g_pos.append((s1,s2,Directions.STOP,(0,False),(s1,s2),p_prob))
            else:
                next_g_pos.append((x1,y1,pos,(scared,b),(s1,s2),p_prob))

        return next_g_pos,collide


    def ghost_rpicky(self,current,ghost,g_pos):
        """
        For a given ghost, get the probability of a move and the position of the ghost on the next turn.
        New facing is also the move the ghost has done.
        :param current: the current PacMan position node
        :param ghost: Id of the ghost
        :param g_pos: ghost positions and orientation (x,y,facing,(scared_Timer,realx,realy),start,p_prob)
        :return: list of (probability, (newX, newY, newFacing))
        """
        x, y, pos,(scared,b),start,p_prob = g_pos[ghost]

        node = self.nodes_array[x][y].getNeighborByDirection(pos)
        if scared > 0 and b and node is not None:
            scared = max(scared-1,0)
            return [(1,(node.i,node.j,pos,(scared,not b),start,p_prob))]
        if not scared and b:
            g_pos[ghost] = (x, y, pos,(scared,False),start,p_prob)

        list_moves = list()
        if p_prob[0] != 0:
            list_moves.extend([(p_prob[0]*prob,pos) for prob,pos in self.ghost_cc_left(current,ghost,g_pos)])
        if p_prob[1] != 0:
            list_moves = [(p_prob[1]*prob,pos) for prob,pos in self.ghost_greedy(current,ghost,g_pos)]
        if p_prob[2] != 0:
            list_moves.extend([(p_prob[2]*prob,pos) for prob,pos in self.random_Move(current,ghost,g_pos)])

        true_list_moves = {pos:0 for prob,pos in list_moves}

        for prob,pos in list_moves:
            true_list_moves[pos] += prob

        ls = [(true_list_moves[pos],pos) for pos in true_list_moves]

        return ls


    def ghost_greedy(self,current,ghost,g_pos):
        """
        For a given ghost, get the probability of a move and the position of the ghost on the next turn.
        New facing is also the move the ghost has done.
        :param current: the current PacMan position node
        :param ghost: Id of the ghost
        :param g_pos: ghost positions and orientation (x,y,facing,scared_Timer,start,p_prob)
        :return: list of (probability, (newX, newY, newFacing))
        """
        x, y, pos ,(st,b),start,p_prob = g_pos[ghost]
        node = self.nodes_array[x][y].getNeighborByDirection(pos)
        if st > 0 and b and node is not None:
            st = max(st-1,0)
            return [(1,(node.i,node.j,pos,(st,not b),start,p_prob))]

        rpos = Actions.reverseDirection(pos)
        node = self.nodes_array[x][y]
        priority = [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]
        node_neig = [node.up, node.left, node.down, node.right]
        prio = list()
        nodes = list()
        for i in range(len(node_neig)):
            if node_neig[i]:
                prio.append(priority[i])
                nodes.append((node_neig[i].i,node_neig[i].j))

        if len(prio) > 1 and rpos != Directions.STOP:
            for i in range(len(prio)):
                if prio[i] == rpos:
                    del prio[i]
                    del nodes[i]
                    break

        arg = min if st == 0 else max
        dist_list = [manhattanDistance(pos, (current.i,current.j)) for pos in nodes]
        ptr_val = arg(dist_list)
        pos = []
        st = st - 1 if st - 1 > 0 else 0
        for i in range(len(dist_list)):
            if dist_list[i] == ptr_val:
                if not st:
                    pos.append((nodes[i][0],nodes[i][1],prio[i],(0,False),start,p_prob))
                else:
                    pos.append((x,y,prio[i],(st,True),start,p_prob))

        pos_p = 1./len(pos)
        return [(pos_p,possibility) for possibility in pos]

    def ghost_randy(self,current,ghost,g_pos):
        """
        For a given ghost, get the probability of a move and the position of the ghost on the next turn.
        New facing is also the move the ghost has done.
        :param current: the current PacMan position node
        :param ghost: Id of the ghost
        :param g_pos: ghost positions and orientation (x,y,facing,scared_Timer,start,p_prob)
        :return: list of (probability, (newX, newY, newFacing))
        """
        x, y, pos ,(st,b),start,p_prob = g_pos[ghost]
        node = self.nodes_array[x][y].getNeighborByDirection(pos)
        if st > 0 and b and node is not None:
            st = max(st-1,0)
            return [(1,(node.i,node.j,pos,(st,not b),start,p_prob))]

        list_moves = [(0.5*prob,pos) for prob,pos in self.ghost_greedy(current,ghost,g_pos)]

        list_moves.extend([(0.25*prob,pos) for prob,pos in self.ghost_cc_left(current,ghost,g_pos)])

        list_moves.extend([(0.25*prob,pos) for prob,pos in self.random_Move(current,ghost,g_pos)])

        true_list_moves = {pos:0 for _,pos in list_moves}

        for prob,pos in list_moves:
            true_list_moves[pos] += prob

        ls = [(true_list_moves[pos],pos) for pos in true_list_moves]

        return ls

    def random_Move(self,current,ghost,g_pos):
        """
        For a given ghost, get the probability of a move and the position of the ghost on the next turn.
        New facing is also the move the ghost has done.
        :param current: the current PacMan position node
        :param ghost: Id of the ghost
        :param g_pos: ghost positions and orientation (x,y,facing,scared_Timer,start,p_prob)
        :return: list of (probability, (newX, newY, newFacing))
        """
        x, y, pos ,(st,b),start,p_prob = g_pos[ghost]
        node = self.nodes_array[x][y].getNeighborByDirection(pos)
        if st > 0 and b and node is not None:
            st = max(st-1,0)
            return [(1,(node.i,node.j,pos,(st,not b),start,p_prob))]

        rpos = Actions.reverseDirection(pos)
        node = self.nodes_array[x][y]
        priority = [Directions.NORTH, Directions.WEST, Directions.SOUTH, Directions.EAST]
        node_neig = [node.up, node.left, node.down, node.right]
        prio = []
        nodes = []
        for i in range(len(node_neig)):
            if node_neig[i]:
                prio.append(priority[i])
                nodes.append(node_neig[i])
        if len(prio) > 1 and rpos != Directions.STOP:
            for i in range(len(prio)):
                if prio[i] == rpos:
                    del prio[i]
                    del nodes[i]
                    break

        st = st - 1 if st - 1 > 0 else 0
        if not st:
            return [(1.0/len(nodes),(nodes[ptr].i,nodes[ptr].j,prio[ptr],(0,False),start,p_prob)) for ptr in range(len(nodes))]
        else:
            return [(1.0/len(nodes),(x,y,prio[ptr],(st,True),start,p_prob)) for ptr in range(len(nodes))]

    def ghost_cc_left(self,current, ghost, g_pos):
        """
        For a given ghost, get the probability of a move and the position of the ghost on the next turn.
        New facing is also the move the ghost has done.
        :param current: the current PacMan position node
        :param ghost: Id of the ghost
        :param g_pos: ghost positions and orientation (x,y,facing,scared_Timer,start,p_prob)
        :return: list of (probability, (newX, newY, newFacing))
        """
        x, y, facing ,(st,b),start,p_prob = g_pos[ghost]
        node = self.nodes_array[x][y].getNeighborByDirection(facing)
        if st > 0 and b and node is not None:
            st = max(st-1,0)
            return [(1,(node.i,node.j,facing,(st,not b),start,p_prob))]

        node = self.nodes_array[x][y]
        st = st - 1 if st - 1 > 0 else 0
        if not st:
            if node.left and (facing != Directions.EAST or len(node.neighbors) == 1):
                return [(1.0, (node.left.i, node.left.j, Directions.WEST,(0,False),start,p_prob))]
            if node.down and (facing != Directions.NORTH or len(node.neighbors) == 1):
                return [(1.0, (node.down.i, node.down.j, Directions.SOUTH,(0,False),start,p_prob))]
            if node.right and (facing != Directions.WEST or len(node.neighbors) == 1):
                return [(1.0, (node.right.i, node.right.j, Directions.EAST,(0,False),start,p_prob))]
            if node.up and (facing != Directions.SOUTH or len(node.neighbors) == 1):
                return [(1.0, (node.up.i, node.up.j, Directions.NORTH,(0,False),start,p_prob))]
        else:
            if node.left and (facing != Directions.EAST or len(node.neighbors) == 1):
                return [(1.0, (node.left.i, node.left.j, Directions.WEST,(st,True),start,p_prob))]
            if node.down and (facing != Directions.NORTH or len(node.neighbors) == 1):
                return [(1.0, (node.down.i, node.down.j, Directions.SOUTH,(st,True),start,p_prob))]
            if node.right and (facing != Directions.WEST or len(node.neighbors) == 1):
                return [(1.0, (node.right.i, node.right.j, Directions.EAST,(st,True),start,p_prob))]
            if node.up and (facing != Directions.SOUTH or len(node.neighbors) == 1):
                return [(1.0, (node.up.i, node.up.j, Directions.NORTH,(st,True),start,p_prob))]


    def terminal_state(self, node, g_pos):
        """
        Checks whether or not we are in terminal state. EG pacman is the same position as a ghost
        :param node:
        :param g_pos:
        :return:
        """
        if self.nb_food == 0:
            return True
        x = node.i
        y = node.j
        for pos in g_pos:
            x1, y1, _, (scared,_),_,_ = pos
            if x == x1 and y == y1 and scared == 0:
                return True
        return False

    def heuristic_val(self, node, g_pos, goal_depth, nb_food,nb_ghost, depth):
        """
        Heuristic for the score of a expect minimax branch, based off many parameters
        :param node: current node
        :param g_pos: ghost postions
        :param atGoal: Whether or not, pacman made it to his goal
        :param nb_food: Number of foods pacman ate on the way to the goal
        :param depth: depth left in the minimax
        :return:
        """
        ret = 0
        if node.has_food:
            ret += 10 # food at that position
        x = node.i
        y = node.j
        for pos in g_pos:
            x1, y1, _, (scared,_),_,_ = pos
            if x == x1 and y == y1:
                if scared == 0:
                    ret -= 1000000  # ghost at that position causing termination
                else:
                    ret += 200

        if goal_depth >= 0:
            ret += goal_depth*50
        else:
            if self.maxDepth - depth < len(self.moves):
                ret -= dist(self.moves[self.maxDepth - depth],node)
            else:
                ret -= dist(self.goal, node)
        ret += nb_food * 10
        ret += 200*nb_ghost

        return ret