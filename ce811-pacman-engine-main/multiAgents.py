

# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)

    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    #print("Best: "+str(bestScore))
    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    successorGameState.getLegalActions()
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    return successorGameState

  def scoreEvaluationFunction(currentGameState):
    """This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
    return currentGameState.getScore()



class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

# class HillClimberAgent(Agent):
#     
#     def registerInitialState(self, state):
#       self.actionList = [];
#       for i in range(0,5):
#         self.actionList.append(Directions.STOP);
#       return;

#     def getAction(self, state):
#         #gameEvaluation =[]
#         temState=currstate=state
#         currmax = gameEvaluation(currstate,temState)
#         bestAction=self.actionList[0]
#         firstIteration=True
#         flag=False
#         loseFlag=False
#         possibleAct = temState.getAllPossibleActions();
        
#         while True:
#             temState=state
            

#             for i in range(0,len(self.actionList)):
#                 ''' For the first action sequence we dont need to change the action'''
#                 if firstIteration:
#                     self.actionList[i] = possibleAct[random.randint(0,len(possibleAct)-1)]
#                     firstIteration=False
#                 else: 
#                     ''' Here we change the action for the sequence, probability of changing=0.5'''
#                     if random.randint(0,1)==1:
#                         self.actionList[i] = possibleAct[random.randint(0,len(possibleAct)-1)]
            
#             for i in range(0,len(self.actionList)):
#                 temState = temState.generatePacmanSuccessor(self.actionList[i])
#                 if temState==None:
#                     flag=True
#                     break
#                 if temState.isLose():
#                     loseFlag=True
#                     break
#                 elif temState.isWin():
#                     return self.actionList[0]
#             if flag:
#                 break
#             if loseFlag:
#                 loseFlag=False
#                 continue
#             if gameEvaluation(currstate,temState)>currmax:
#                 currmax=gameEvaluation(currstate,temState)
#                 bestAction=self.actionList[0]

#         return bestAction
        
    
# class GeneticAgent(Agent):
#     
#     def registerInitialState(self, state):
#         self.actionList = [];
#         for i in range(0,5):
#             self.actionList.append(Directions.STOP);
#         return;

#     def getAction(self, state):
#         popList = []
#         #gameEvaluation = []
#         temState=currstate=state
#         currmax = gameEvaluation(currstate,temState)
#         bestAction=self.actionList[0]
#         possibleAct = temState.getAllPossibleActions();
#         flag=True
#         loseFlag=False
        
#         for i in range(0,8):
#             for j in range(0, len(self.actionList)):
#                 self.actionList[j] = possibleAct[random.randint(0, len(possibleAct) - 1)]
#             popList.append(self.actionList[:])

#         while flag:
#             pop = []
#             for i in range(0,8):
#                 temState=state
#                 for j in range(0, len(self.actionList)):
#                     temState = temState.generatePacmanSuccessor(popList[i][j])
#                     if temState==None:
#                         flag=False
#                         break
#                     if temState.isLose():
#                         break
#                     elif temState.isWin():
#                         return self.actionList[0]
#                 if flag==False:
#                     break
#                 pop.append((popList[i][:],gameEvaluation(currstate,temState)))
            
#             if flag==False:
#                 break
#             pop.sort(key = lambda x:x[1], reverse=True)
#             if pop[0][1]>currmax:
#                 bestAction=pop[0][0][0]
            
#             popList=[]
#             for i in range (0,4):
#                 rank1,rank2 = self.pairSelect()
#                 Xchromosome = pop[rank1][0][:]
#                 Ychromosome = pop[rank2][0][:]
#                 crossover = random.randint(0,100)
                
#                 if random.randint(0,100)<=70:
#                     children1=[]
#                     children2=[]
#                     for j in range(0,5):
#                         if random.randint(0,1)==1:
#                             children1.append(Xchromosome[j])
#                             children2.append(Ychromosome[j])
#                         else:
#                             children1.append(Ychromosome[j])
#                             children2.append(Xchromosome[j])
#                     popList.append(children1)
#                     popList.append(children2)
#                 else:
#                     popList.append(Xchromosome[:])
#                     popList.append(Ychromosome[:])
                    
#             for i in range(0,8): 
#                 if random.randint(0, 100) <= 10:
#                     popList[i][random.randint(0,4)] = possibleAct[random.randint(0,len(possibleAct)-1)]
                    
#         return bestAction
    
    


#     def pairSelect(self):
#         x=random.randint(1,55)
#         y=random.randint(1,55)
        
#         x = self.calculateProb(x)
#         y = self.calculateProb(y)
#         while x==y:
#             y=random.randint(1,55)
#             y = self.calculateProb(y)
#         return x,y
    
#     def calculateProb(self,number): 
#         if number==1:
#             return 7
#         elif number<=3:
#             return 6
#         elif number <=6:
#             return 5
#         elif number<=10:
#             return 4
#         elif number <=15:
#             return 3
#         elif number<=21:
#             return 2
#         elif number <=28:
#             return 1
#         elif number<=36:
#             return 0
#         elif number<=45:
#             return 0
#         elif number<=55:
#             return 0



class FinalGAAgent(Agent):

  def __init__(self, index=0, chromosome = [0,1,2,3,4,5]):
    super().__init__(index)
    self.chromosome = chromosome

    
  """def Play(self,gameState):
    bScore=0
    chromosome =[]
    scores=[]
    moves = self.getAction(self, gameState) 
    # Choose one of the best actions
    for action in moves:
      score = self.evaluationFunction(gameState,action)
      if bScore<score:
        bScore=score
        chromosome.append(action)
        scores.append(bScore)

    print("ch: "+str(chromosome))
    pass"""
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    
    
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)  
    
    
    
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    #print("Best: "+str(bestScore))
    "Add more of your code here if you want to"
    # return successorGameState.getScore()



  # for rule in self.chromosome:
  #   if rule == 0:
  #   #safety first, take direction furtherst from all ghosts unless they're scared
  #     nearestGhostDist = min(nearestGhostDist,manhattanDistance((Xghost, Yghost), newPos))

  #   elif rule == 1:
  #   #prioritize fruits if ghosts > 10 away
  #     nearestFoodDist = min(nearestFoodDist, manhattanDistance(food, newPos))

  #   elif rule == 2:
  #   #always prioritize fruit
  #     food = 

  #   elif rule == 3:
  #   #prioritize power pellets (big fruits to kill ghosts), or catch ghosts
  #     pass

  #   elif rule == 4:
  #   #take third/fourth direction if ghosts are closing in from other directions 
  #     pass

  #   elif rule == 5:
  #   #act randomly
  #     pass

    return legalMoves[chosenIndex]

  # def mutation(offspringCrossover, percentMutation):
  #   vectSize = offspringCrossover.shape[0]
  #   numberOfMutations=(int)(vectSize*percentMutation)
  #   randidx = np.random.choice(range(vectSize), size=(numberOfMutations), replace=False)
  #   np.put(offspringCrossover, randidx, np.random.uniform(0, 1))
  #   return offspringCrossover


  # def crossover(p1, p2):
  #   # layer_size=[26, 52, 26, 13, 4]
  #   global weights
  #   global nGenerations
  #   newChildreWeights = {}
  #   for child in range(nGenerations):
  #       childNet = []        
  #       for layer in range(4):
  #           Matrix1 = weights[p1][layer]
  #           Matrix2 = weights[p2][layer]
  #           Flat1 = Matrix1.flatten()
  #           Flat2 = Matrix2.flatten()
  #           matrixSize = Matrix1.shape[0]*Matrix1.shape[1]
  #           randidx = np.random.choice(
  #               range(matrixSize), size=(matrixSize), replace=False)
  #           #print(f"my matrix size {matrixSize}")
  #           halfSize=(int)(matrixSize/2)
  #           np.put(Flat1, randidx[halfSize:], 0)
  #           np.put(Flat2, randidx[:halfSize], 0)
  #           crossed = Flat1 + Flat2
  #           mutated = mutation(crossed, 0.01)
  #           childNet.append(crossed.reshape(
  #               Matrix1.shape[0], Matrix1.shape[1]))
  #       newChildreWeights[child] = childNet
  #   weights = newChildreWeights



  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    foods = newFood.asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    nearestGhostDist = 10
    for ghostState in newGhostStates:
      Xghost, Yghost = ghostState.getPosition()
      Xghost = int(Xghost)
      Yghost = int(Yghost)
      if ghostState.scaredTimer == 0:
        nearestGhostDist = min(nearestGhostDist,manhattanDistance((Xghost, Yghost), newPos))

    nearestFoodDist = 10
    for food in foods:
      nearestFoodDist = min(nearestFoodDist, manhattanDistance(food, newPos))
    if not foods:
      nearestFoodDist = 0
    return successorGameState.getScore() - 7 / (nearestGhostDist + 1) - nearestFoodDist / 3



   

  def scoreEvaluationFunction(currentGameState):
    """This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
    return currentGameState.getScore()



