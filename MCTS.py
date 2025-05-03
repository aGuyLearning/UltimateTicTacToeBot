import copy
import numpy as np
import random
import math
import GameRepresentationFunctional as GameRepresentation

# state = (global_state x, global state o, local state x, local state o, currentplayer, currentboard, winner)

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state        # Game state or environment state 
        self.parent = parent        # Parent node
        self.children = []         # Child nodes
        self.visits = 0            # Number of times node was visited
        self.value = 0            # Accumulated value from simulations
        self.untried_actions = GameRepresentation.getPossibleMoves(*state)  # Actions not yet expanded, move array [(0,1)]
        self.action = None         # Action that led to this node
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """Check if the node represents a terminal state"""
        return self.state[-1] != GameRepresentation.Status.RUNNING
        
    def expand(self):
        """Expand a node by creating a new child"""
        action = self.untried_actions.pop()
        next_state = copy.deepcopy(self.state)
        next_state = GameRepresentation.move(*next_state, action) 
        child_node = MCTSNode(next_state, parent=self)
        child_node.action = action
        self.children.append(child_node)
        return child_node
    
    def getUCB(self, exploration_weight=1.4):
        """Calculate Upper Confidence Bound (UCB) for this node"""
        if self.visits == 0:
            return float('inf')
        
        top_level = self
        if top_level.parent is not None:
            top_level = top_level.parent
        
        return self.value / self.visits + exploration_weight * math.sqrt(math.log(top_level.visits) / self.visits)
    
    def best_child(self, exploration_weight=1.4):
        """Select the best child according to UCT (Upper Confidence Bound for Trees)"""
        weights = np.array([child.getUCB(exploration_weight) for child in self.children])
        return self.children[np.argmax(weights)]
    
    def rollout(self):
        """Perform a random simulation from this node's state"""
        current_state = copy.deepcopy(self.state)
        while current_state[-1] is None:  # Implement is_terminal for your problem
            action = GameRepresentation.getPossibleMoves(*current_state) #current_state.random_action()  # Implement random action selection
            action = random.choice(action)
            #action = sorted(action, key=lambda m: self.move_priority(m))
            #action = action[0]
            current_state = GameRepresentation.move(*current_state, action)
        return get_reward(current_state)

    def move_priority(self, move):
        x, y = move
        x = x % 3
        y = y % 3
        if (x, y) == (1, 1):
            return 0  # center
        elif (x, y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 1  # corners
        else:
            return 2  # edges
    
    def backpropagate(self, reward):
        """Backpropagate the simulation result"""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

def get_reward(state):
    x_turn = state[4]
    if x_turn:
        if state[-1] == GameRepresentation.Status.WIN_X:
            return 3
        elif state[-1] == GameRepresentation.Status.WIN_O:
            return 0
        else:
            return 1
    else:
        if state[-1] == GameRepresentation.Status.WIN_O:
            return 3
        elif state[-1] == GameRepresentation.Status.WIN_X:
            return 0
        else:
            return 1
        

class MCTS:
    def __init__(self, initial_state, iteration_limit=1000):
        self.root = MCTSNode(initial_state)
        self.iteration_limit = iteration_limit
    
    def search(self):
        """Run the MCTS algorithm"""
        for _ in range(self.iteration_limit):
            node = self.select(self.root)
            reward = self.simulate(node)
            node.backpropagate(reward)
        return self.get_best_action()
    
    def select(self, node):
        """Select a node to expand"""
        while not node.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child()
            else:
                return node.expand()
        return node
    
    def simulate(self, node):
        """Run a simulation from the given node"""
        if node.is_terminal():
            return get_reward(node.state)
        return node.rollout()
    
    def get_best_action(self):
        """Get the best action based on current search results"""
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda x: x.visits).action
    
    
    def update_root(self, action):
        """Advance the tree to the child node corresponding to the taken action"""
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None
                return
        raise ValueError("Action not found in root's children")
    

    def print_tree(self, node, prefix=""):
        children = node.children
        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            print(prefix + connector + str(str(child.value) + "/" + str(child.visits)))
            extension = "    " if i == len(children) - 1 else "│   "
            #self.print_tree(child, prefix + extension)