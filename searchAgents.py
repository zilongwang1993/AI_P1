# searchAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
This file contains all of the agents that can be selected to
control Pacman.  To select an agent, use the '-p' option
when running pacman.py.  Arguments can be passed to your agent
using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a searchFunction=depthFirstSearch

Commands to invoke other search strategies can be found in the
project description.

Please only change the parts of the file you are asked to.
Look for the lines that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the
project description for details.

Good luck and happy searching!
"""
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
from game import Grid

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search algorithm for a
    supplied search problem, then returns actions to follow that path.

    As a default, this agent runs DFS on a PositionSearchProblem to find location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game board. Here, we
        choose a path to the goal.  In this phase, the agent should compute the path to the
        goal and store it in a local variable.  All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in registerInitialState).  Return
        Directions.STOP if there is no further action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded

        "*** YOUR CODE HERE ***"
        self.startingState = (self.startingPosition, (0, 0, 0, 0))
        self.verticalDist = top - 1
        self.horizontalDist = right - 1

    def getStartState(self):
        "Returns the start state (in your state space, not the full Pacman state space)"
        "*** YOUR CODE HERE ***" 
        return self.startingState

    def isGoalState(self, state):
        "Returns whether this search state is a goal state of the problem"
        "*** YOUR CODE HERE ***"
        return sum(state[1]) == 4

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            x,y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                cornerStates = list(state[1])
                if (nextx, nexty) in self.corners:
                    foodIndex = self.corners.index((nextx, nexty))
                    cornerStates[foodIndex] = 1
                nextState = ((nextx, nexty), tuple(cornerStates))
                cost = 1
                successors.append( ( nextState, action, cost) )


        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem; i.e.
    it should be admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    numFoodsLeft = 4 - sum(state[1])
    foodIndices = [x for x in range(4) if state[1][x] == 0]
    if numFoodsLeft == 0:
        return 0
    elif numFoodsLeft == 1:
        return util.manhattanDistance(state[0], corners[foodIndices[0]])
    elif numFoodsLeft == 2:
        return (min([util.manhattanDistance(state[0], corners[i]) for i in foodIndices])
            + util.manhattanDistance(corners[foodIndices[0]], corners[foodIndices[1]]))
    elif numFoodsLeft == 3:
        if 0 in foodIndices and 3 in foodIndices:
            opposites = [0, 3]
        else:
            opposites = [1, 2]
        return (min([util.manhattanDistance(state[0], corners[i]) for i in opposites])
            + problem.verticalDist + problem.horizontalDist)
    elif numFoodsLeft == 4:
        return (min([util.manhattanDistance(state[0], corners[i]) for i in foodIndices])
            + problem.verticalDist + problem.horizontalDist + min(problem.verticalDist, problem.horizontalDist))
            
    return 0 # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem



def generateDistMaps(problem):
    """
    cache the distances from each food to each position
    """
    gameState = problem.startingGameState
    wallMap = gameState.getWalls()
    foodMap = gameState.getFood()
    height = foodMap.height
    width = foodMap.width
    foods = [(x1, y1) for x1 in range(width) for y1 in range(height) if foodMap[x1][y1]]
    distMaps = {}
    
    for xFood, yFood in foods:
        distMap = [[None]*height for x in range(width)]
        # distMaps[(xFood, yFood)][x][y] should be the distance in the maze from (xFood, yFood) to (x, y)
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        # do BFS to fill the distMap
        bfsQueue = util.Queue()
        bfsQueue.push((xFood, yFood, 0))
        while not bfsQueue.isEmpty():
            x, y, dist = bfsQueue.pop()
            if distMap[x][y] is None:
                distMap[x][y] = dist
                for dx, dy in offsets:
                    if not wallMap[x + dx][y + dy]:
                        bfsQueue.push((x + dx, y + dy, dist + 1))
        distMaps[(xFood, yFood)] = distMap
    return distMaps

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a
    Grid (see game.py) of either True or False. You can call foodGrid.asList()
    to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    if 'distMaps' not in problem.heuristicInfo:
        problem.heuristicInfo['distMaps'] = generateDistMaps(problem)
    distMaps = problem.heuristicInfo['distMaps']
    
    x, y = state[0]
    foodMap = state[1]
    height = foodMap.height
    width = foodMap.width
    remainingFoods = [(x1, y1) for x1 in range(width) for y1 in range(height) if foodMap[x1][y1]]
    maxdist = 0
    for food in remainingFoods:
        maxdist = max(maxdist, distMaps[food][x][y])
    return maxdist

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        "Returns a path (a list of actions) to the closest dot, starting from gameState"
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return search.aStarSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
      A search problem for finding a path to any food.

      This search problem is just like the PositionSearchProblem, but
      has a different goal test, which you need to fill in below.  The
      state space and successor function do not need to be changed.

      The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
      inherits the methods of the PositionSearchProblem.

      You can use this search problem to help you fill in
      the findPathToClosestDot method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test
        that will complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return self.food[x][y]

##################
# Mini-contest 1 #
##################

def findNearestBadNode(startNodeNum, nodes, edges, nodeEdges, nodeGood, edgeChosen):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    heap = util.PriorityQueue()
    nodeDists = [None] * nodeCount
    nodeDists[startNodeNum] = 0
    nodePaths = [[] for x in range(nodeCount)]
    nodePaths[startNodeNum] = []
    heap.push(startNodeNum, 0)
    while not heap.isEmpty():
        i = heap.pop()
        for edgeNum, edgeEnd, neighbor, cost in nodeEdges[i]:
            if edgeNum in nodePaths[i]:
                continue
            if edgeChosen[edgeNum]:
                nextDist = nodeDists[i] - cost
            else:
                nextDist = nodeDists[i] + cost
            if nodeDists[neighbor] is None or nodeDists[neighbor] > nextDist:
                nodeDists[neighbor] = nextDist
                nodePaths[neighbor] = nodePaths[i][:]
                nodePaths[neighbor].append(edgeNum)
                heap.push(neighbor, nextDist)
    nearests = []
    paths = []
    costs = []
    for i in range(nodeCount):
        if nodeGood[i] or i == startNodeNum:
            continue
        nearests.append(i)
        paths.append(nodePaths[i])
        costs.append(nodeDists[i])
    ind = costs.index(min(costs))
    return nearests[ind], paths[ind], costs[ind]

def generateNodeInfo(nodes, edges, startNode):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs = [0] * nodeCount
    nodeEdges = [[] for x in range(nodeCount)]
    for j in range(edgeCount):
        if edges[j] is None:
            continue
        s, e, c = edges[j]
        nodeDegs[s] += 1
        nodeDegs[e] += 1
        nodeEdges[s].append((j, 0, e, c))
        nodeEdges[e].append((j, 1, s, c))
    return nodeDegs, nodeEdges

def calcMinPathLength(nodes, edges, startNode):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs, nodeEdges = generateNodeInfo(nodes, edges, startNode)
    nodeGood = [(deg % 2 == 0) for deg in nodeDegs]
    nodeGood[startNode] = not nodeGood[startNode]
    edgeChosen = [False] * edgeCount
    while sum(nodeGood) < nodeCount - 1:
        nums = []
        nearests = []
        paths = []
        costs = []
        for i in range(nodeCount):
            if nodeGood[i]:
                continue
            nearest, path, cost = findNearestBadNode(i, nodes, edges, nodeEdges, nodeGood, edgeChosen)
            nums.append(i)
            nearests.append(nearest)
            paths.append(path)
            costs.append(cost)
        ind = costs.index(min(costs))
        i = nums[ind]
        nearest = nearests[ind]
        path = paths[ind]
        cost = costs[ind]
        #print "Connecting %d and %d with cost %d" % (i, nearest, cost)
        nodeGood[i] = not nodeGood[i]
        nodeGood[nearest] = not nodeGood[nearest]
        for j in path:
            edgeChosen[j] = not edgeChosen[j]
    length = 0
    #print nodeGood.index(False) #210
    for j in range(edgeCount):
        if edges[j] is None:
            continue
        if edgeChosen[j]:
            length += 2 * edges[j][2]
        else:
            length += edges[j][2]
    return length, edgeChosen

def convertMapToGraph(wallMap, foodMap, startPos):
    width = wallMap.width
    height = wallMap.height
    nodes = []
    nodeNumForPos = {}
    nodePositions = []
    edges = []
    startNode = None
    for x in range(width):
        for y in range(height):
            if (x, y) == startPos:
                nodeNumForPos[(x, y)] = startNode = len(nodes)
                nodes.append(((x, y),))
            elif foodMap[x][y]:
                #if nodeCount == 210:
                #    print (startPos, x, y)
                nodeNumForPos[(x, y)] = len(nodes)
                nodes.append(((x, y),))
    for x in range(width):
        for y in range(height - 1):
            if (x, y) in nodeNumForPos and (x, y + 1) in nodeNumForPos:
                edges.append((nodeNumForPos[(x, y)], nodeNumForPos[(x, y + 1)], 1))
    for x in range(width - 1):
        for y in range(height):
            if (x, y) in nodeNumForPos and (x + 1, y) in nodeNumForPos:
                edges.append((nodeNumForPos[(x, y)], nodeNumForPos[(x + 1, y)], 1))
    return nodes, edges, startNode

def printChosen(nodes, edges, startNode, edgeChosen, wallMap):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    width = wallMap.width
    height = wallMap.height
    nodeChosen = [False] * nodeCount
    for j in range(edgeCount):
        if edges[j] is None:
            continue
        if edgeChosen[j]:
            nodeChosen[edges[j][0]] = True
            nodeChosen[edges[j][1]] = True
    g = Grid(width, height)
    for x in range(width):
        for y in range(height):
            if wallMap[x][y]:
                g[x][y] = '%'
            else:
                g[x][y] = ' '
    for i in range(nodeCount):
        if nodes[i] is None:
            continue
        x, y = nodes[i][0]
        if nodeChosen[i]:
            g[x][y] = '+'
        else:
            g[x][y] = '.'
    print str(g)

def generateCapMap(nodes, edges, startNode, edgeChosen, wallMap):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs, nodeEdges = generateNodeInfo(nodes, edges, startNode)
    width = wallMap.width
    height = wallMap.height
    nodeChosen = [False] * nodeCount
    for j in range(len(edges)):
        if edgeChosen[j]:
            nodeChosen[edges[j][0]] = True
            nodeChosen[edges[j][1]] = True
    capMap = Grid(width, height)
    for x in range(width):
        for y in range(height):
            capMap[x][y] = 0
    for i in range(nodeCount):
        if nodes[i] is None:
            continue
        x, y = nodes[i][0]
        if nodeDegs[i] > 2:
            capMap[x][y] = 2
        elif nodeChosen[i] and nodeDegs[i] == 2:
            capMap[x][y] = 2
        else:
            capMap[x][y] = 1
    return capMap

def consolidateCliques(nodes, edges, startNode):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs, nodeEdges = generateNodeInfo(nodes, edges, startNode)
    nodeColors = [None] * nodeCount # not None = inside the clique of this color
    colorCount = 0
    # from every 4-deg nodes, fill into 4-deg and 3-deg nodes
    for i in range(nodeCount):
        if nodes[i] is None:
            continue
        if nodeDegs[i] != 4 or nodeColors[i] is not None:
            continue
        currentColor = colorCount
        colorCount += 1
        queue = util.Queue()
        queue.push(i)
        found = 0
        while not queue.isEmpty():
            i2 = queue.pop()
            if nodeDegs[i2] < 3 or nodeColors[i2] is not None:
                continue
            nodeColors[i2] = currentColor
            found += 1
            for edgeNum, edgeEnd, neighbor, cost in nodeEdges[i2]:
                queue.push(neighbor)
        # don't create single-node cliques
        if found <= 1:
            nodeColors[i] = None
            colorCount -= 1
    # fill 1-deg and 2-deg nodes that connects to the same color on its both ends
    for i in range(nodeCount):
        if nodes[i] is None:
            continue
        if nodeDegs[i] == 2:
            color = nodeColors[nodeEdges[i][0][2]] # color of destination node of the first edge from i
            color2 = nodeColors[nodeEdges[i][1][2]] # color of destination node of the second edge from i
            if color is None or color2 is None or color != color2:
                continue
        elif nodeDegs[i] == 1:
            color = nodeColors[nodeEdges[i][0][2]]
            if color is None:
                continue
        else:
            continue
        nodeColors[i] = color
    # generate cliques
    #  a hyper node is either an original node or a clique
    hyperNodes = [None] * nodeCount
    hyperEdges = edges[:]
    for i in range(nodeCount):
        if nodes[i] is not None:
            hyperNodes[i] = (nodes[i][0], i, None)
    for color in range(colorCount):
        hyperNodes.append((None, None, color)) #temporarily set its position to None
        
    hyperStartNode = startNode # should not be a part of a clique
    cliques = [([None] * nodeCount, [None] * edgeCount) for c in range(colorCount)]
    for i in range(nodeCount):
        if nodes[i] is None:
            continue
        color = nodeColors[i]
        if color is not None:
            cliqueNodes, cliqueEdges = cliques[color]
            # add or modify the node and the edges connect to it
            hyperNodes[i] = None
            if hyperNodes[nodeCount + color][0] is None:
                hyperNodes[nodeCount + color] = list(hyperNodes[nodeCount + color])
                hyperNodes[nodeCount + color][0] = nodes[i][0]
                hyperNodes[nodeCount + color] = tuple(hyperNodes[nodeCount + color])
            cliqueNodes[i] = nodes[i]
            for edgeNum, edgeEnd, neighbor, cost in nodeEdges[i]:
                if nodeColors[neighbor] is None:
                    hyperEdges[edgeNum] = list(hyperEdges[edgeNum])
                    hyperEdges[edgeNum][edgeEnd] = nodeCount + color
                    # TODO: estimate a cost for this hyperedge
                    hyperEdges[edgeNum] = tuple(hyperEdges[edgeNum])
                else:
                    if hyperEdges[edgeNum] == None:
                        continue
                    cliqueEdges[edgeNum] = edges[edgeNum]
                    hyperEdges[edgeNum] = None
    return cliques, hyperNodes, hyperEdges, hyperStartNode
    
def spliceEdges(nodes, edges, startNode):
    # i.e. remove 2-deg nodes
    nodes = nodes[:]
    edges = edges[:]
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs, nodeEdges = generateNodeInfo(nodes, edges, startNode)
    edgePaths = [([] if e is not None else None) for e in edges]
    for i in range(nodeCount):
        if nodes[i] is None or nodeDegs[i] != 2 or i == startNode:
            continue
        edgeNum1, edgeEnd1, neighbor1, cost1 = nodeEdges[i][0]
        edgeNum2, edgeEnd2, neighbor2, cost2 = nodeEdges[i][1]
        if neighbor1 == i or neighbor2 == i or edgeNum1 == edgeNum2:
            continue
        nodes[i] = None
        nodeEdges[i] = []
        edges[edgeNum2] = None
        cost = cost1 + cost2
        edges[edgeNum1] = (neighbor1, neighbor2, cost)
        if edgeEnd1 != 1:
            edgePaths[edgeNum1].reverse()
        if edgeEnd2 != 0:
            edgePaths[edgeNum2].reverse()
        edgePaths[edgeNum1] = edgePaths[edgeNum1] + [i] + edgePaths[edgeNum2]
        edgePaths[edgeNum2] = None
        ind = nodeEdges[neighbor1].index((edgeNum1, 1 - edgeEnd1, i, cost1))
        nodeEdges[neighbor1][ind] = (edgeNum1, 0, neighbor2, cost)
        ind = nodeEdges[neighbor2].index((edgeNum2, 1 - edgeEnd2, i, cost2))
        nodeEdges[neighbor2][ind] = (edgeNum1, 1, neighbor1, cost)
    return edgePaths, nodes, edges, startNode

def tryRemoveEdge(num, edgeRemoved, nodes, edges, startNode):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs, nodeEdges = generateNodeInfo(nodes, edges, startNode)
    colorCount = 0
    nodeColors = [None] * nodeCount
    for i in range(nodeCount):
        if nodes[i] is None or nodeColors[i] is not None:
            continue
        color = colorCount
        colorCount += 1
        queue = util.Queue()
        queue.push(i)
        while not queue.isEmpty():
            i2 = queue.pop()
            if nodeColors[i2] is not None:
                continue
            nodeColors[i2] = color
            for edgeNum, edgeEnd, neighbor, cost in nodeEdges[i2]:
                if edgeRemoved[edgeNum] or edgeNum == num:
                    continue
                queue.push(neighbor)
    return (colorCount <= 1)

def iterateEdges(edgeChosen, edgeRemoved, nodes, edges, startNode):
    nodeCount = len(nodes)
    edgeCount = len(edges)
    nodeDegs, nodeEdges = generateNodeInfo(nodes, edges, startNode)
    edgeUsage = [0] * len(edges)
    maxUsage = [(2 if edgeChosen[j] else 1 if edges[j] is not None else 0) for j in range(edgeCount)]
    current = startNode
    linkedEdges = []
    while True:
        bestNum = None
        bestEnd = None
        for edgeNum, edgeEnd, neighbor, cost in nodeEdges[current]:
            if edgeUsage[edgeNum] >= maxUsage[edgeNum]:
                continue
            bestNum = edgeNum
            bestEnd = edgeEnd
            if edgeUsage[edgeNum] <= maxUsage[edgeNum] - 2:
                break
        if bestNum is None:
            break
        linkedEdges.append((bestNum, bestEnd))
        if edgeRemoved[bestNum]:
            edgeUsage[bestNum] += 2 # "removed" edges are added to the array only once, but this item represents a round trip on the "removed" edge
        else:
            edgeUsage[bestNum] += 1
            current = edges[bestNum][1 - bestEnd]
    #print "iterate edges: " + str(sum(maxUsage) - sum(edgeUsage)) + " edges left"
    return linkedEdges

def findShortestPath(nodes, edges, startNode):
    cliques, hyperNodes, hyperEdges, hyperStartNode = consolidateCliques(nodes, edges, startNode)
    reducedEdgePaths, reducedNodes, reducedEdges, reducedStartNode = spliceEdges(hyperNodes, hyperEdges, hyperStartNode)
    reducedLength, reducedEdgeChosen = calcMinPathLength(reducedNodes, reducedEdges, reducedStartNode)
    reducedEdgeRemoved = [False] * len(reducedEdges)
    for j in range(len(reducedEdges)):
        if reducedEdges[j] is None:
            continue
        if not reducedEdgeChosen[j]:
            continue
        if tryRemoveEdge(j, reducedEdgeRemoved, reducedNodes, reducedEdges, reducedStartNode):
            #print "Removed an edge of cost " + str(reducedEdges[j][2])
            reducedEdgeRemoved[j] = True
        else:
            #print "Couldn't remove an edge of cost " + str(reducedEdges[j][2])
            pass
    linkedEdges = iterateEdges(reducedEdgeChosen, reducedEdgeRemoved, reducedNodes, reducedEdges, reducedStartNode)
    hyperPath = [reducedStartNode]
    for edgeNum, edgeEnd in linkedEdges:
        p = reducedEdgePaths[edgeNum][:]
        if edgeEnd == 1:
            p.reverse()
        if reducedEdgeRemoved[edgeNum]:
            # append the path and its returning path, skipping the end point
            hyperPath = hyperPath + p
            p.reverse()
            p = p + [reducedEdges[edgeNum][edgeEnd]]
            hyperPath = hyperPath + p[1:]
        else:
            # append the path and the end point
            hyperPath = hyperPath + p + [reducedEdges[edgeNum][1 - edgeEnd]]
    #print " Hyperpath contains " + str(len([i for i in hyperPath if i >= len(nodes)])) + " clique nodes"
    return [i for i in hyperPath if i < len(nodes)]

# Graph = (nodes: Node[], edges: Edge[], startNode: NodeNum)
# Node = (pos: (x, y), )
# Edge = (start: NodeNum, end: NodeNum, cost)
# NodeEdge = (edgenum: EdgeNum, edgeend: 0/1, neighbor: NodeNum, cost)[]
# Clique = (nodes: Node[], edges: Edge[])
# HyperNode = (pos: (x, y), original: NodeNum?, clique: NodeNum?)

"""
class ContestProblem:
    def __init__(self, startingGameState):

        self.wallMap = self.walls = startingGameState.getWalls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.startingGameState = startingGameState
        self.foodMap = startingGameState.getFood()
        pos = startingGameState.getPacmanPosition()
        visitMap = self.foodMap.copy()
        for x in range(self.width):
            for y in range(self.height):
                visitMap[x][y] = 0
        self.start = (pos, visitMap)
        #self.capMap = generateCapMap(nodes, edges, startNode, edgeChosen, self.wallMap)
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information
        
    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        if self._expanded % 200 == 0:
            print self._expanded
            print state[1].count(1)
            print state[1].count(2)
            print state[1].count(3)
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if (not self.walls[nextx][nexty]) and state[1][nextx][nexty] < self.capMap[nextx][nexty]:
                visitMap = state[1].copy()
                visitMap[nextx][nexty] += 1
                successors.append( ( ((nextx, nexty), visitMap), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        "Returns the cost of a particular sequence of actions.  If those actions include an illegal move, return 999999"
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            #dx, dy = Actions.directionToVector(action)
            #x, y = int(x + dx), int(y + dy)
            #if self.walls[x][y]:
            #    return 999999
            cost += 1
        return cost

def contestHeuristic(state, problem):
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    if 'distMaps' not in problem.heuristicInfo:
        problem.heuristicInfo['distMaps'] = generateDistMaps(problem)
    distMaps = problem.heuristicInfo['distMaps']
    
    x, y = state[0]
    visitMap = state[1]
    height = visitMap.height
    width = visitMap.width
    remainingFoods = [(x1, y1) for x1 in range(width) for y1 in range(height) if problem.foodMap[x1][y1] and visitMap[x1][y1] == 0]
    maxdist = 0
    #for food in remainingFoods:
    #    maxdist = max(maxdist, distMaps[food][x][y])
    return max(maxdist, len(remainingFoods)) * 1.0
"""

def graphTest(state):
    #print calcMinPathLength(10, [(0,3,1),(1,1,1),(1,2,1),(2,3,1),(3,4,1),(2,5,1),
    #    (3,6,1),(6,7,1),(5,8,10),(5,8,20),(6,9,1)], 8)
    nodes, edges, startNode = convertMapToGraph(startingGameState.getWalls(), startingGameState.getFood(), startingGameState.getPacmanPosition())
    #print len(nodes)
    #print len(edges)
    #print startNode
    length, edgeChosen = calcMinPathLength(nodes, edges, startNode)
    #print length
    cliques, hyperNodes, hyperEdges, hyperStartNode = consolidateCliques(nodes, edges, startNode)
    reducedEdgePaths, reducedNodes, reducedEdges, reducedStartNode = spliceEdges(hyperNodes, hyperEdges, hyperStartNode)
    printChosen(reducedNodes, reducedEdges, reducedStartNode, [False] * len(reducedEdges), startingGameState.getWalls())
    print len([n for n in reducedNodes if n is not None])
    print len([e for e in reducedEdges if e is not None])
    print "Sum of edge costs:"
    print sum([e[2] for e in reducedEdges if e is not None])
    print sum([e[2] for e in hyperEdges if e is not None])
    printChosen(hyperNodes, hyperEdges, hyperStartNode, edgeChosen, startingGameState.getWalls())
    print length
    for cliqueNodes, cliqueEdges in cliques:
        printChosen(cliqueNodes, cliqueEdges, startNode, edgeChosen, startingGameState.getWalls())
        print len([n for n in cliqueNodes if n is not None])
        print len([e for e in cliqueEdges if e is not None])
    hyperLength, hyperEdgeChosen = calcMinPathLength(hyperNodes, hyperEdges, hyperStartNode)
    printChosen(hyperNodes, hyperEdges, hyperStartNode, hyperEdgeChosen, startingGameState.getWalls())
    print hyperLength
    quit()

class ApproximateSearchAgent(Agent):
    "Implement your contest entry here.  Change anything but the class name."

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"
        #problem = ContestProblem(state)
        #func = getattr(search, 'aStarSearch')
        #path = func(problem, contestHeuristic)
        #print path
        #print len(path)
        #print problem.getCostOfActions(path)
        nodes, edges, startNode = convertMapToGraph(state.getWalls(), state.getFood(), state.getPacmanPosition())
        path = findShortestPath(nodes, edges, startNode)
        actions = []
        for i in range(len(path) - 1):
            pos = nodes[path[i]][0]
            nextpos = nodes[path[i + 1]][0]
            if nextpos == (pos[0] + 1, pos[1]):
                action = Directions.EAST
            elif nextpos == (pos[0] - 1, pos[1]):
                action = Directions.WEST
            elif nextpos == (pos[0], pos[1] + 1):
                action = Directions.NORTH
            elif nextpos == (pos[0], pos[1] - 1):
                action = Directions.SOUTH
            else:
                print "Cannot find action from " + str(pos) + " to " + str(nextpos)
                action = None
            actions.append(action)
        print "Found a path with " + str(len(actions)) + " actions"
        self.actions = actions
        self.time = 0

    def getAction(self, state):
        """
        From game.py:
        The Agent will receive a GameState and must return an action from
        Directions.{North, South, East, West, Stop}
        """
        "*** YOUR CODE HERE ***"
        action = self.actions[self.time]
        self.time += 1
        return action

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(search.bfs(prob))
