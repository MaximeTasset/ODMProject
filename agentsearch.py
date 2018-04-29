from pacman import Directions
from game import Agent
import sys
# XXX: You should complete this class for Step 1
# apparently in 2D arrays containing the map, up/down are reversed
from search import SearchProblemData, astar, dist, aStarHeuristic, Graph


class Agentsearch(Agent):
    def __init__(self, index=0, time_eater=40, g_pattern=-1):
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
        self.moves = []
        self.graph = None
        super().__init__(index)

    def reset(self):
        self.moves = []
        self.graph = None

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
        if self.graph is None:
            self.graph = Graph(state)
        else:
            self.graph.update_graph(state)

        if len(self.moves) == 0:

            nodes_list = self.graph.nodes_list
            nodes_array = self.graph.nodes_array
            # start node is the node where pacman is positioned
            start = nodes_array[self.graph.posX][self.graph.posY]

            # this method finds the 2 food nodes furthest from each other and
            # sends pacman to the closest of the 2
            food_nodes = []
            for node in nodes_list:
                if node.has_food:
                    food_nodes.append(node)
            end = None
            middle = None
            distance = - 1
            for i in range(0, len(food_nodes)):
                node1 = food_nodes[i]
                for j in range(i, len(food_nodes)):
                    node2 = food_nodes[j]
                    if distance < dist(node1, node2):
                        end = node1
                        middle = node2
                        distance = dist(node1, node2)

            if dist(middle, start) > dist(end, start):
                middle, end = end, middle

            # this method tries to go to the closest node from pacman
            distance = sys.maxsize
            end2 = None
            for node in nodes_list:
                if node.has_food and node is not start and self.graph.dist(
                        node) < distance:
                    distance = self.graph.dist(node)
                    end2 = node

            # since we dont recalculate the path at each iteration, we
            # combine both methods
            # If there's a really close point compared to the middle point
            # we go there instead
            # This helps to avoid skipping points on one side then going to
            # another and forcing pacman
            # to go all the way back
            if dist(end2, start) * 3 < dist(middle, start):
                middle = end2
            cell = self.graph.nodes_array[self.graph.posX][self.graph.posY]
            if middle is not None:
                # look for a path to end node. It ends early if it finds food
                # closer than the end node
                problem = SearchProblemData(nodes_list, start, middle)
                path, _ = astar(problem, aStarHeuristic)
                # path = dfs(problem)
                if path is not None:
                    self.moves = path
                    ret = path[0]
                    del self.moves[0]
                    negb_cell = cell.getNeighborByDirection(ret)
                    if self.graph.nb_food == 1 and negb_cell.has_food:
                        self.reset()
                    return ret

            return Directions.STOP
        else:
            ret = self.moves[0]
            del self.moves[0]
            negb_cell = cell.getNeighborByDirection(ret)
            if self.graph.nb_food == 1 and negb_cell.has_food:
                self.reset()
            return ret
