# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
from game import Directions

class Node:
    """
    This class contains information on a specific position on the map, using
    pointers to connect to other positions
    """
    def __init__(self, i, j, has_food,has_capsule=False):
        """
        Inits the class
        :param i: X coordinate
        :param j: Y coordinate
        :param has_food: Whether or not there is food
        :param has_capsule: Whether or not there is  a capsule
        """
        self.i = i
        self.j = j
        self.neighbors = []
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.has_food = has_food
        self.has_capsule = has_capsule

    def getNeighborDirection(self, neighbor):
        """
        From a node determines if it's a neighbor and its direction relative to the
        current node
        :param neighbor: node that essentially should be neighbor
        :return:
        """
        if neighbor == self.up:
            return Directions.NORTH
        if neighbor == self.down:
            return Directions.SOUTH
        if neighbor == self.left:
            return Directions.WEST
        if neighbor == self.right:
            return Directions.EAST
        return None

    def getNeighborByDirection(self, direction):
        """
        Returns the neighbor based on the direction
        :param direction: from the Directions class
        :return: concerned neighbor
        """
        if direction == Directions.NORTH:
            return self.up
        if direction == Directions.SOUTH:
            return self.down
        if direction == Directions.WEST:
            return self.left
        if direction == Directions.EAST:
            return self.right
        if direction == Directions.STOP:
            return self
        return None

    def __str__(self):
        return "(" + str(self.i) + " " + str(self.j) + ")"

    def __repr__(self):
        sum = len(self.neighbors)

        return "(" + str(self.i) + " " + str(self.j) + " " + str(sum) + ")"


class SearchProblemData:
    """
    This class contains the data concerning the path from point a to point b
    """

    def __init__(self, nodes, start, end):
        """
        Inits the class
        :param nodes: The entire graph
        :param start: Start node
        :param end: End/Destination Node
        """
        self.nodes = nodes
        self.start = start
        self.goal = end

        #dictionaries + lists to contain info on our states
        self.open_set = []
        self.closed_set = []
        self.came_from = {}
        self.path_from = {}
        self.g_score = {}
        self.f_score = {}


    def getStartState(self):
        """
        Gets the start state
        :return: start state
        """
        return self.start


    def isGoalState(self, state):
        """
        Checks if current state is the goal state
        :param state: current state
        :return: True/False
        """
        return state is self.goal


    def getSuccessors(self, state):
        """
        Get Neighbor states of current state with the movement associated
        :param state: Current state
        :return: A list of of a trio of (neigbor node, direction, weight)
        """
        ret = []
        for node in state.neighbors:
            if node.i == state.i + 1:
                tmp = Directions.EAST
            elif node.i == state.i - 1:
                tmp = Directions.WEST
            elif node.j == state.j + 1:
                tmp = Directions.NORTH
            else:
                tmp = Directions.SOUTH
            tmp = (node, tmp, self.g_score[node])
            ret.append(tmp)
        return ret


    def getCostOfActions(self, actions):
        """
        Gets the cost of the actions
        :param actions: a list containing the actions
        :return: the length of  the list
        """
        return len(actions)

    def reconstruct_path(self, current):
        """
        Reconstructs the path from the start state to the current state
        This works if the current state has already been visited
        :param current: the current state
        :return: list of movements, list of nodes that it goes through
        """
        total_path = [current]
        total_movements = [self.path_from[current]]
        while current in self.came_from:
            current = self.came_from[current]
            total_path.append(current)
            total_movements.append(self.path_from[current])
        total_movements = total_movements[:-1]
        total_path = total_path[:-1]
        return list(reversed(total_movements)), list(reversed(total_path))


def dist_between(node1, node2):
    """
    calculate Distance between 2 nodes
    :param node1: pos1
    :param node2: pos2
    :return:
    """
    x1 = node1.i
    x2 = node2.i
    y1 = node1.j
    y2 = node2.j
    return abs(x1 - x2) + abs(y1 - y2)


class Graph:
    """
    This class creates a graph from the current game state
    and positions. It contains a list of state nodes and a 2D array
    of state nodes depending on how you want access the graph
    """

    def __init__(self, state):
        """
        Inits the graph
        :param state: Game state containing up to date info on the game
        """
        self.state = state
        # get basic information on grid and positions
        self.height = state.data.layout.height
        self.width = state.data.layout.width
        self.posX, self.posY = state.getPacmanPosition()
        self.nb_food = 0
        self.nb_capsule = 0
        self.nb_nodes = 0
        self.mean_choices = 0
        self.nodes_list, self.nodes_array = self.create_graph()


    def create_graph(self):
        """
        Creates the graph, nodes stored in list and in 2D array
        :return: list of nodes, array of nodes
        """
        self.nb_nodes = 0
        self.mean_choices = 0
        wall_map = self.state.getWalls()
        # create a graph of all the possible positions
        # stored in nodes_list
        nodes_list = []
        # nodes array is to link nodes together after creation
        # it's a 2D array and for each node, it creates link if there is a node in a direction
        nodes_array = []
        # Create nodes with empty links
        for i in range(0, self.width):
            tmp = []
            for j in range(0, self.height):
                if wall_map[i][j] is False:
                    if self.state.hasFood(i, j):
                        node = Node(i, j, True)
                        self.nb_food += 1
                    elif (i,j) in self.state.getCapsules():
                        node = Node(i, j, False,True)
                        self.nb_capsule += 1
                    else:
                        node = Node(i, j, False)
                    self.nb_nodes += 1
                    nodes_list.append(node)
                    tmp.append(node)
                else:
                    tmp.append(0)
            nodes_array.append(tmp)
        # setup node links
        for i in range(0, self.width):
            for j in range(0, self.height):
                if nodes_array[i][j] != 0:
                    if self.is_in(i, j + 1) and nodes_array[i][j + 1] != 0:
                        nodes_array[i][j].up = nodes_array[i][j + 1]
                        nodes_array[i][j].neighbors.append(nodes_array[i][j + 1])
                        self.mean_choices += 1
                    if self.is_in(i + 1, j) and nodes_array[i + 1][j] != 0:
                        nodes_array[i][j].right = nodes_array[i + 1][j]
                        nodes_array[i][j].neighbors.append(nodes_array[i + 1][j])
                        self.mean_choices += 1
                    if self.is_in(i, j - 1) and nodes_array[i][j - 1] != 0:
                        nodes_array[i][j].down = nodes_array[i][j - 1]
                        nodes_array[i][j].neighbors.append(nodes_array[i][j - 1])
                        self.mean_choices += 1
                    if self.is_in(i - 1, j) and nodes_array[i - 1][j] != 0:
                        nodes_array[i][j].left = nodes_array[i - 1][j]
                        nodes_array[i][j].neighbors.append(nodes_array[i - 1][j])
                        self.mean_choices += 1
        if self.nb_nodes:
            self.mean_choices /= self.nb_nodes
        return nodes_list, nodes_array

    def update_graph(self, state):
        self.state = state
        self.posX, self.posY = state.getPacmanPosition()
        if self.nodes_array[self.posX][self.posY].has_food:
            self.nb_food -= 1
        self.nodes_array[self.posX][self.posY].has_food = False
        if self.nodes_array[self.posX][self.posY].has_capsule:
            self.nb_capsule -= 1
        self.nodes_array[self.posX][self.posY].has_capsule = False

    def is_in(self, i, j):
        """
        Checks if a coordinate is in the 2D array
        :param i: width coordinate
        :param j: height coordinate
        :return: True/False
        """
        return (i >= 0 and j >= 0) and (i < self.width and j < self.height)

    def dist(self, node):
        """
        Calculates distance between pacman and a node
        :param node: current node
        :return: Distance between pacman and the node
        """
        x = node.i
        y = node.j
        return abs(x - self.posX) + abs(y - self.posY)


def aStarHeuristic(state, problem=None):
    """
    Heuristic used for the A* alogorithm
    :param state: Current state node
    :param problem: class containing SearchProblemData
    :return: the distance between the current node and goal node
    """
    if problem is None :
        return 0
    node1 = state
    node2 = problem.goal
    x1 = node1.i
    x2 = node2.i
    y1 = node1.j
    y2 = node2.j
    return abs(x1 - x2) + abs(y1 - y2)


def dist(node1, node2):
    """
    Simple distance between 2 nodes
    :param node1: 1st state node
    :param node2: 2nd state node
    :return: distance between nodes
    """
    x1 = node1.i
    x2 = node2.i
    y1 = node1.j
    y2 = node2.j
    return abs(x1 - x2) + abs(y1 - y2)


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())


    BIG FLAW with depth first :
        Since the neighbor nodes are put in a certain order (LEFT then RIGHT then UP then DOWN),
        If there are multiple paths to a food position from one turn to another, pacman may decide
        to switch paths to get to the food (since it doesnt look for the shortest path but instead the 1st path)
        Because of that, pacman will be alternating between the two nodes forever because they give different paths
    """
    start = problem.getStartState()

    goal = problem.goal
    nodes = problem.nodes
    problem.came_from = {}
    problem.path_from = {start: Directions.STOP}
    problem.open_set = [start]
    problem.g_score = {}
    problem.h_score = {}
    for node in nodes:
        problem.g_score.update({node: sys.maxsize})
        problem.g_score.update({start: 0})

    while len(problem.open_set) > 0:
        node = problem.open_set.pop()
        if node not in problem.h_score:
            problem.h_score[node] = 1
            neighbors = problem.getSuccessors(node)
            for neighbor in neighbors:
                neighbor_node, move, _ = neighbor
                if neighbor_node not in problem.h_score:
                    problem.came_from[neighbor_node] = node
                    problem.path_from[neighbor_node] = move
                    problem.open_set.append(neighbor_node)

    return problem.reconstruct_path(goal)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    A* star algorithm, The idea is to search positions with the lowest heuristic
    value first. For Pacman, this means go through the points that are in a straight
    line from him to the destination food first.
    :param problem: SearchDataProblem Object
    :param heuristic: heuristic used to measure distance
    :return: path and movements chosen for the next few turns (see reconstruct_path from SearchDataProblem)
    """
    start = problem.getStartState()
    nodes = problem.nodes
    problem.closed_set = []

    problem.open_set = [start]

    problem.came_from = {}
    problem.path_from = {start: Directions.STOP}

    problem.g_score = {}
    for node in nodes:
        problem.g_score.update({node: sys.maxsize})
        problem.g_score.update({start: 0})

    problem.f_score = {}
    for node in nodes:
        problem.f_score.update({node: sys.maxsize})
        problem.f_score[start] = heuristic(start, problem)

    while len(problem.open_set) != 0:

        current = problem.open_set[0]
        current_val = problem.f_score[current]
        current_index = 0
        for i in range(1, len(problem.open_set)):
            tmp = problem.open_set[i]
            if problem.f_score[tmp] < current_val:
                current = problem.open_set[i]
                current_val = problem.f_score[current]
                current_index = i

        if problem.isGoalState(current) or current.has_food:
            problem.goal = current
            return problem.reconstruct_path(current)

        del problem.open_set[current_index]
        problem.closed_set.append(current)

        neighbors = problem.getSuccessors(current)
        for neighbor in neighbors:
            neighbor_node, move, cost = neighbor
            if neighbor_node in problem.closed_set:
                continue
            if neighbor_node not in problem.open_set:
                problem.open_set.append(neighbor_node)

            tentative_gscore = problem.g_score[current] + dist_between(current, neighbor_node)
            if tentative_gscore >= cost:
                continue
            problem.came_from[neighbor_node] = current
            problem.path_from[neighbor_node] = move
            problem.g_score[neighbor_node] = tentative_gscore
            problem.f_score[neighbor_node] = problem.g_score[neighbor_node] + heuristic(neighbor_node, problem)
    return None

# Abbreviations
dfs = depthFirstSearch
astar = aStarSearch