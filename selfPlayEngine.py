import GameRepresentationFunctional 
from MCTS import MCTS
from NNet import UltimateTTTNet

def best_move(state):
    mcts = MCTS(state, 10000)
    mcts.search()
    best_action = mcts.get_best_action()
    if best_action:
        return best_action
    else:
        print("no action found")

if __name__ == "__main__":
    nnet = UltimateTTTNet()
    mcts = MCTS(GameRepresentationFunctional.INITIAL_STATE, 25, nnet)
    for i in range(100):
        mcts.search()
        best_action = mcts.get_best_action()
        if best_action:
            mcts.update_root(mcts.get_best_action())
        else:
            print(GameRepresentationFunctional.stringRep(*mcts.root.state))
            break