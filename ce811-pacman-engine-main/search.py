

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

def graphSearch(problem, front):
    front.push((problem.getStartState(), []))
    explore = []
    while not front.isEmpty():
        state, actions = front.pop()
        if problem.isGoalState(state):
            return actions
        if state not in explore:
            explore.append(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                if successor not in explore:
                    nextActions = actions + [action]
                    successorNode = (successor, nextActions)
                    front.push(successorNode)
    return []


def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    front = util.Stack()
    return graphSearch(problem, front)


def breadthFirstSearch(problem):
    def costFunc(state, actions):
        return len(actions)
    front = util.PriorityQueueWithFunction(costFunc)
    front.push((problem.getStartState(), []))
    return graphSearch(problem, front)


def uniformCostSearch(problem):
    def costFunc(state, actions):
        return problem.getCostOfActions(actions)
    front = util.PriorityQueueWithFunction(costFunc)
    return graphSearch(problem, front)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    def costFunc(state, actions):
        return problem.getCostOfActions(actions) + heuristic(state, problem)
    front = util.PriorityQueueWithFunction(costFunc)
    return graphSearch(problem, front)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


