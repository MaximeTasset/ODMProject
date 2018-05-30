from expectminimax import ExpectMiniMax
from pacman import Directions
from game import Agent
import numpy as np
import sys
# XXX: You should complete this class for Step 2
from search import SearchProblemData, astar, dist, aStarHeuristic, Graph

class Agentghost(Agent):
    """
    Similar to agentSearch except that it just takes the closest food
    destination and it looks to handle ghosts
    """

    def __init__(self, index=0, time_eater=40, g_pattern=0, allSame=True):
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
        - `allSame`: True if rpickyghosts follow the same pattern,
                     False otherwise
        """
        self.g_pattern = g_pattern

        self.g_patterns_prob = None
        self.g_pattern_predict = {}

        self.moves = []
        self.moves_nodes = []
        self.depth = 8
        self.max_d = 7
        self.allSame = allSame
        self.problem = None
        self.goal = None
        self.graph = None
        super().__init__(index)
        self.known = {}
        self.start = None

    def reset(self):
        self.g_patterns_prob = None
        self.g_pattern_predict = {}

        self.moves = []
        self.moves_nodes = []

        self.problem = None
        self.goal = None
        self.graph = None
        self.known = {}


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
        try:
            if self.start is None:
                self.start = tuple(
                    state.data.agentStates[0].start.getPosition())

            direction = state.data.agentStates[0].getDirection()
            pos = tuple(state.getPacmanPosition())
            if self.graph is None or (direction == Directions.STOP and pos == self.start):
                self.reset()
                self.graph = Graph(state)
                ch = self.graph.mean_choices
                if ch <= 2.3:
                    mult = 1.
                elif ch > 2.3 and ch <= 3:
                    mult = 0.75
                else:
                    mult = 0.5
                if self.g_pattern >= 2:
                    self.depth = int(5 * mult)
                elif self.g_pattern == 0:
                    self.depth = int(8 * mult)
                else:
                    self.depth = int(8 * mult)

            else:
                self.graph.update_graph(state)

            ret = Directions.STOP

            nodes_list = self.graph.nodes_list
            nodes_array = self.graph.nodes_array
            # start node is the node where pacman is positioned
            start = nodes_array[self.graph.posX][self.graph.posY]

            if len(self.moves) == 0:

                # this method tries to go to the closest node from pacman
                distance = sys.maxsize
                end2 = None
                for node in nodes_list:
                    if node.has_food and node is not start and self.graph.dist(
                            node) < distance:
                        distance = self.graph.dist(node)
                        end2 = node

                middle = end2
                if middle is not None:
                    # look for a path to end node. It ends early if it finds food
                    # closer than the end node
                    self.problem = SearchProblemData(nodes_list, start, middle)
                    self.goal = middle
                    path, path_nodes = astar(self.problem, aStarHeuristic)
                    # path = dfs(problem)
                    if path is not None:
                        self.moves = path
                        self.moves_nodes = path_nodes

            # get ghost info
            ghosts = state.getGhostStates()
            pos = []
            if self.g_patterns_prob is None and len(ghosts):
                self.g_patterns_prob = np.zeros((len(ghosts), 3))
                if self.g_pattern == 3:
                    self.g_patterns_prob[:, :] = 1. / 3
                else:
                    self.g_patterns_prob[:, self.g_pattern] = 1.
            else:
                pass

            for i, ghost in enumerate(ghosts):
                realx, realy = ghost.getPosition()
                x, y = int(realx), int(realy)
                if not ghost.scaredTimer and self.g_pattern == 3 \
                        and i in self.g_pattern_predict:
                    for pattern in self.g_pattern_predict[i]:
                        p_moves = self.g_pattern_predict[i][pattern]
                        if (x, y) in p_moves:
                            prob = p_moves[(x, y)]
                            if not self.allSame:
                                self.g_patterns_prob[i, pattern] *= prob
                            else:
                                self.g_patterns_prob[:, pattern] *= prob
                        else:
                            if not self.allSame:
                                self.g_patterns_prob[i, pattern] = 0
                            else:
                                self.g_patterns_prob[:, pattern] = 0
                    del self.g_pattern_predict[i]

            for i, ghost in enumerate(ghosts):
                realx, realy = ghost.getPosition()
                x, y = int(realx), int(realy)
                try:
                    p_prob = self.g_patterns_prob[i, :] / \
                        sum(self.g_patterns_prob[i, :])
                    self.g_patterns_prob[i, :] = p_prob[:]
                except BaseException:
                    p_prob = np.zeros(3)
                    p_prob[:] = self.g_patterns_prob[i, :] = 1. / 3

                for p, prob in enumerate(p_prob):
                    if prob > 0.92:
                        self.g_patterns_prob[i, :] = 0
                        self.g_patterns_prob[i, p] = 1
                        break

                facing = ghost.getDirection()
                b = int(x) != realx or int(y) != realy
                pos.append(
                    (int(x),
                     int(y),
                        facing,
                        (ghost.scaredTimer,
                         b),
                        ghost.start.pos,
                        tuple(p_prob)))
            # do minimax if there are ghosts close enough
            doMinMax = False
            cell = self.graph.nodes_array[self.graph.posX][self.graph.posY]
            for coord in pos:
                x, y, _, _, _, _ = coord
                ghost_node = nodes_array[x][y]
                if dist(start, ghost_node) < self.depth:
                    doMinMax = True

            if doMinMax:
                minmax = ExpectMiniMax(
                    self.graph,
                    start,
                    self.goal,
                    self.g_pattern,
                    self.moves_nodes,
                    self.depth)
                # update class and do expectminimax
                _, moves = minmax.expect(start, self.depth, True, pos)
                move = moves[len(moves) - 1]
                if move != self.moves[0]:
                    self.moves = []
                    self.moves_nodes = []
                    if self.g_pattern == 3:
                        self.predict_ghost(move, pos)

                    negb_cell = cell.getNeighborByDirection(move)
                    if self.graph.nb_food == 1 and negb_cell.has_food:
                        self.reset()
                    return move

            ret = self.moves[0]
            del self.moves[0]
            del self.moves_nodes[0]
            if self.g_pattern == 3:
                self.predict_ghost(ret, pos)
            negb_cell = cell.getNeighborByDirection(ret)
            if self.graph.nb_food == 1 and negb_cell.has_food:
                self.reset()

        except (IndexError,AttributeError):
            self.reset()
            raise IndexError
        return ret

    def predict_ghost(self, move, g_pos):
        """
        This method looks to predict the ghost type in the case of rpicky based
        on previous moves done by the ghost
        :param move: Directions move done
        :param g_pos: ghost positional information
        :return:
        """
        cell = self.graph.nodes_array[self.graph.posX][self.graph.posY]
        start = cell.getNeighborByDirection(move)
        eM = ExpectMiniMax(
            self.graph,
            start,
            None,
            self.g_pattern,
            None,
            0)
        patterns = [eM.ghost_cc_left, eM.ghost_greedy, eM.ghost_randy]
        if start.has_capsule:
            return
        for i, (x, y, _, (scared, _), _, _) in enumerate(g_pos):
            if scared > 0:
                continue
            # only compute for undetermined and uneaten ghosts
            if (start.i, start.j) != (x, y):
                if len(np.nonzero(self.g_patterns_prob[i, :])[0]) > 1:
                    self.g_pattern_predict[i] = {}
                    for j, pattern in enumerate(patterns):
                        if self.g_patterns_prob[i, j]:
                            predict = {(x1, y1): prob
                                       for prob, (x1, y1, _, _, _, _)
                                       in pattern(start, i, g_pos)}
                            self.g_pattern_predict[i][j] = predict
                elif i not in self.known:
                    print("ghost {} is {}".format(
                        i, self.g_patterns_prob[i, :]))
                    if not self.g_patterns_prob[i,
                                                2] and self.depth < self.max_d:
                        self.depth += 1
                    self.known[i] = True
