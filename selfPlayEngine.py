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
    import torch
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
    # nnet = UltimateTTTNet()
    # mcts = MCTS(GameRepresentationFunctional.INITIAL_STATE, 25, nnet)
    # for i in range(100):
    #     mcts.search()
    #     best_action = mcts.get_best_action()
    #     if best_action:
    #         mcts.update_root(mcts.get_best_action())
    #     else:
    #         print(GameRepresentationFunctional.stringRep(*mcts.root.state))
    #         break