import copy
import numpy as np
import random
import math
import GameRepresentationFunctional as GameRepresentation

# state = (global_state x, global state o, local state x, local state o, currentplayer, currentboard, winner)

class MCTSNode:
    def __init__(self, state, nnet, parent=None):
        self.state = state        # Game state or environment state 
        self.parent = parent        # Parent node
        self.children = []         # Child nodes
        self.visits = 0            # Number of times node was visited
        self.value = 0            # Accumulated value from simulations
        self.untried_actions = GameRepresentation.getPossibleMoves(*state)  # Actions not yet expanded, move array [(0,1)]
        self.action = None         # Action that led to this node
        self.nnet = nnet
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """Check if the node represents a terminal state"""
        return self.state[-1] != GameRepresentation.Status.RUNNING
        
    def expand(self):
        """Expand a node by creating a new child"""
        action = self.untried_actions.pop()
        next_state = copy.deepcopy(self.state)
        next_state = GameRepresentation.move(*next_state, *action) 
        child_node = MCTSNode(next_state, parent=self, nnet=self.nnet)
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
            valid_moves = GameRepresentation.getPossibleMoves(*current_state)
           # Get policy probabilities ONLY for valid moves
            policy_probs, _ = self.nnet.predict(current_state, valid_moves)

            # Filter probabilities for only valid moves
            valid_probs = []
            valid_indices = []
            for x, y in valid_moves:
                index = y * 9 + x  # Convert (x,y) to flat index
                valid_probs.append(policy_probs[index])
                valid_indices.append(index)
        
            # Normalize the probabilities (sum to 1)
            prob_sum = sum(valid_probs)
            if prob_sum <= 0:  # Handle edge case
                valid_probs = [1/len(valid_moves)] * len(valid_moves)
            else:
                valid_probs = [p/prob_sum for p in valid_probs]
            
            # Choose action by weighted random selection
            chosen_idx = random.choices(range(len(valid_moves)), weights=valid_probs, k=1)[0]
            chosen_move = valid_moves[chosen_idx]
            current_state = GameRepresentation.move(*current_state, *chosen_move)
        return get_reward(current_state) 
    
    def backpropagate(self, reward):
        """Backpropagate the simulation result"""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(1 - reward)

def get_reward(state):
    """Get the reward for the current state"""
    if state[-1] == "D":
        return 0
    elif state[-1] == 1:
        return 1
    else:
        return -1
        

class MCTS:
    def __init__(self, initial_state, iteration_limit=1000, nnet=None):
        self.root = MCTSNode(initial_state, nnet)
        self.iteration_limit = iteration_limit
        self.nnet = nnet
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
    
    def get_action_probabilities(self, num_samples=100, temp=1):
        """Get action probabilities based on visit counts"""
        for _ in range(num_samples):
            self.search()
        
        counts = [child.visits for child in self.root.children]
        if temp == 0:
            best_action = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def print_tree(self, node, prefix=""):
        children = node.children
        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            print(prefix + connector + str(str(child.value) + "/" + str(child.visits)))
            extension = "    " if i == len(children) - 1 else "│   "
            #self.print_tree(child, prefix + extension)