import copy
import logging
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import time

import numpy as np
import torch
from tqdm import tqdm

from Arena import Arena
import GameRepresentationFunctional
from MCTS_NEW import MCTSNodeLess
from NNet import UltimateTTTNet, state_to_tensor
from Utils import AverageMeter
import torch.optim as optim

log = logging.getLogger(__name__)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, nnet, device, args):
        self.device =device
        self.nnet = nnet.to(self.device) 
        print(f"Using device: {self.device}")
        self.pnet = self.nnet.__class__().to(self.device)  # create a new instance of the neural network
        self.args = args
        self.mcts = MCTSNodeLess(self.nnet, args["numMCTSSims"])
        self.trainExamplesHistory = []  
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = copy.deepcopy(GameRepresentationFunctional.INITIAL_STATE)
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args["tempThreshold"])

            pi = self.mcts.getActionProb(copy.deepcopy(board), temp=temp)
            trainExamples.append([state_to_tensor(board), self.curPlayer, pi, None])
            sym = GameRepresentationFunctional.get_symmetries(*board) #TODO: get symmetries for pi 
            pi_sym = GameRepresentationFunctional.flip_arr(np.array(pi).tolist())
            for sym_board, sym_pi in zip(sym, pi_sym):
                trainExamples.append([state_to_tensor(sym_board), self.curPlayer, sym_pi, None])

            valid = GameRepresentationFunctional.getPossibleMoves(*board)
            action = np.random.choice(len(pi), p=pi)  # pick an action according to the policy
            
            board = GameRepresentationFunctional.move(*board, valid[action])

            r = board[-1] != None

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args["numIters"] + 1):
            # bookkeeping
            print(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args["maxlenOfQueue"])

                for _ in tqdm(range(self.args["numEps"]), desc="Self Play"):
                    self.mcts = MCTSNodeLess(self.nnet)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args["numItersForTrainExamplesHistory"]:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args["checkpoint"], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args["checkpoint"], filename='temp.pth.tar')
            pmcts = MCTSNodeLess(self.pnet)

            self.train(trainExamples)
            nmcts = MCTSNodeLess(self.nnet)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)))
            pwins, nwins, draws = arena.playGames(self.args["arenaCompare"])

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args["updateThreshold"]:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args["checkpoint"], filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args["checkpoint"], filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args["checkpoint"], filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args["checkpoint"]
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args["load_folder_file"][0], self.args["load_folder_file"][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            print('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True


    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args["epochs"]):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args["batch_size"])

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args["batch_size"])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert boards to batch tensor
                # remove the batch dimension from boards


                boards = torch.stack([b.squeeze(0) for b in boards]).to(self.device)  # Shape: [batch_size, 6, 9, 9]

                # Convert policy and value targets
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)  # Shape: [batch_size, 9]
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).unsqueeze(1).to(self.device)  # Shape: [batch_size, 1]



                # compute output
                out_pi, out_v = nnet(boards)
                l_pi = self.nnet.loss_pi(target_pis, out_pi)
                l_v = self.nnet.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

if __name__ == "__main__":
    args = {
        'numIters': 10,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': 1,          # Number of games moves for MCTS to simulate.
        'arenaCompare': 1,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,             # Upper confidence bound for MCTS exploration.
        'checkpoint': './temp/',
        'load_model': False,
        "epochs": 10,  
        'batch_size': 32,
        'numItersForTrainExamplesHistory': 20,
    }
    device = "cpu"
    nnet = UltimateTTTNet(device=device)
    coach = Coach(nnet, device, args)
    # nnet.load_checkpoint(folder=args["checkpoint"], filename='checkpoint_2.pth.tar')
    start = time.time()

    coach.learn()
