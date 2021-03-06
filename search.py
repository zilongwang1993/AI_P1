# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    dfsStack =util.Stack()
    result=[]
    dfsDict={}
    visited =set()
    lastNode=None
    found=False
    dfsStack.push(problem.getStartState())

    while not dfsStack.isEmpty():
        cur =dfsStack.pop()
        if problem.isGoalState(cur):
            found=True
            lastNode=cur
            break       
        visited.add(cur)
        sucs = problem.getSuccessors(cur)
        for suc in sucs:
            if suc[0] not in visited:
                dfsStack.push(suc[0])
                dfsDict[suc[0]]=(cur,suc[1])
    if found:
        while lastNode in dfsDict:
            temp=dfsDict[lastNode] 
            result.append(temp[1])
            lastNode=temp[0]
        result.reverse()
        return result   
    else:            
        return result




def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    scheduled = {}
    previousStateAndAction = {}
    fringeStates = util.Queue()
    # initialize with the start state
    state = problem.getStartState()
    scheduled[state] = True
    previousStateAndAction[state] = None
    fringeStates.push(state)
    goalState = None
    while not fringeStates.isEmpty():
        # get the oldest element in the fringe
        state = fringeStates.pop()
        if problem.isGoalState(state):
            goalState = state
            break
        # explore the successors of that state and add them to the fringe
        for successorState, action, cost in problem.getSuccessors(state):
            if successorState not in scheduled:
                scheduled[successorState] = True
                previousStateAndAction[successorState] = (state, action)
                fringeStates.push(successorState)
    if goalState is None:
        return None
    # track predecessors to get the path
    path = []
    state = goalState
    while previousStateAndAction[state] is not None:
        state, action = previousStateAndAction[state]
        path.append(action)
    path.reverse()
    return path

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    stepsForFinishedStates = {}
    fringeSteps = util.PriorityQueue()
    # initialize with the start state
    # tuple step := (state, previousStep, action, totalCost)
    step = (problem.getStartState(), None, None, 0.0)
    fringeSteps.push(step, step[3])
    goalStep = None
    while not fringeSteps.isEmpty():
        # As the while loop proceeds, the cost is always increasing
        step = fringeSteps.pop()
        state = step[0]
        if problem.isGoalState(state):
            goalStep = step
            break
        # if this is a fresh state, mark the state as finished
        if state not in stepsForFinishedStates:
            stepsForFinishedStates[state] = step
            # add its neighbors to the heap
            for successorState, action, cost in problem.getSuccessors(state):
                successorStep = (successorState, step, action, step[3] + cost)
                fringeSteps.push(successorStep, successorStep[3])
    if goalStep is None:
        return None
    path = []
    step = goalStep
    while step[1] is not None:
        path.append(step[2])
        step = step[1]
    path.reverse()
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    stepsForFinishedStates = {}
    fringeSteps = util.PriorityQueue() # key = totalCost + heuristic
    # initialize with the start state
    # tuple step := (state, previousStep, action, totalCost)
    state = problem.getStartState()
    step = (state, None, None, 0.0)
    fringeSteps.push(step, step[3] + heuristic(state, problem))
    goalStep = None
    while not fringeSteps.isEmpty():
        # As the while loop proceeds, the value of cost+heuristic is always increasing
        step = fringeSteps.pop()
        state = step[0]
        if problem.isGoalState(state):
            goalStep = step
            break
        # if this is a fresh state, mark the state as finished
        if state not in stepsForFinishedStates:
            stepsForFinishedStates[state] = step
            # add its neighbors to the heap
            for successorState, action, cost in problem.getSuccessors(state):
                successorStep = (successorState, step, action, step[3] + cost)
                fringeSteps.push(successorStep, successorStep[3] + heuristic(successorState, problem))
    if goalStep is None:
        return None
    path = []
    step = goalStep
    while step[1] is not None:
        path.append(step[2])
        step = step[1]
    path.reverse()
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
