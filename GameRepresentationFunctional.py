import copy
import random
import time
import numpy as np
from enum import Enum

SYMMETRY_INDICES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8], #identity
    [6, 3, 0, 7, 4, 1, 8, 5, 2], # rotate 90
    [8, 7, 6, 5, 4, 3, 2, 1, 0], # rotate 180
    [2, 5, 8, 1, 4, 7, 0, 3, 6], # rotate 270
    [2, 1, 0, 5, 4, 3, 8, 7, 6], #flip vertical
    [6, 7, 8, 3, 4, 5, 0, 1, 2], # flip horizontal
    [0, 3, 6, 1, 4, 7, 2, 5, 8], # flip main diagonal
    [8, 5, 2, 7, 4, 1, 6, 3, 0], # flip anti diagonal
]

LOCAL_SYMMETRY_INDICES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8], #identity
    [2, 5, 8, 1, 4, 7, 0, 3, 6], # rotate 270 
    [8, 7, 6, 5, 4, 3, 2, 1, 0], # rotate 180
    [6, 3, 0, 7, 4, 1, 8, 5, 2], # rotate 90
    [2, 1, 0, 5, 4, 3, 8, 7, 6], #flip vertical
    [6, 7, 8, 3, 4, 5, 0, 1, 2], # flip horizontal
    [0, 3, 6, 1, 4, 7, 2, 5, 8], # flip main diagonal
    [8, 5, 2, 7, 4, 1, 6, 3, 0], # flip anti diagonal
]


WIN_MASKS = [
    # horizontal
    0b000000111,
    0b000111000,
    0b111000000,
    # vertical
    0b001001001,
    0b010010010,
    0b100100100,
    #diagonal
    0b100010001,
    0b001010100,
]

class Status(Enum):
    RUNNING = 0
    DRAW = 1
    WIN_X = 2
    WIN_O = 3
    

INITIAL_STATE = (
    0b0, #global_state x
    0b0, # global state o
    0b0, # local state x
    0b0, # local state o
    True, # currentplayer
    9, # 0-8 are the local boards, 9 is free choice
    Status.RUNNING, # winner
)

def move(global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner, move):
        
        if (checkValidMove(global_board_x, global_board_o, board_x, board_o, next_move_board_idx, move)):
            board_idx = move // 9

            if currentPlayer:
                board_x |= (1 << move)
                if checkWin(board_x, board_idx):
                    global_board_x |= (1 << board_idx)
                    if checkWin(global_board_x & ~global_board_o):
                        winner = Status.WIN_X
                        return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)                   
                    elif checkDraw(global_board_x, global_board_o):
                        winner = Status.DRAW
                        return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner) 
            else:
                board_o |= (1 << move)
                if checkWin(board_x, board_idx):
                    global_board_o |= (1 << board_idx)
                    if checkWin(global_board_o & ~global_board_x):
                        winner = Status.WIN_O
                        return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)
                    elif checkDraw(global_board_x, global_board_o):
                        winner = Status.DRAW
                        return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)
                    
            if checkDraw(board_x, board_o, board_idx):
                global_board_x |= (1 << board_idx)
                global_board_o |= (1 << board_idx)
                if checkDraw(global_board_x, global_board_o):
                    winner = Status.DRAW
                    return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)
                
            next_move_board_idx = move % 9
            if isNotPlayableBoard(global_board_x, global_board_o, next_move_board_idx):
                next_move_board_idx = 9
            
            return (global_board_x, global_board_o, board_x, board_o, not currentPlayer, next_move_board_idx, winner)
        return None


#--------------------------------

def checkWin(board, board_idx = 0):
    for mask in WIN_MASKS:
        mask = mask << (board_idx * 9)
        if (board & mask) == mask:
            return True
    return False

def checkDraw(board_x, board_o, board_idx = 0):
    # check if all local boards are full
    mask = 0b111111111 << (board_idx * 9)
    return (((board_x | board_o) & mask) == mask)

def isSetOnBoard(board_x, board_o, move):
    return ((board_x | board_o) & (1 << move))

def isNotPlayableBoard(global_board_x, global_board_o, board_idx):
    # check if the board is playable
    return (global_board_x & (1 << board_idx) or global_board_o & (1 << board_idx))
             
### general functions
def checkValidMove(global_board_x, global_board_o, board_x, board_o, next_move_board_idx, move):
    board_idx = move // 9
    # check taht the board is valid 
    if not ((0 <= board_idx <=9) and (next_move_board_idx == 9 or board_idx == next_move_board_idx)):
        return False
    # check that global board is not set
    if isNotPlayableBoard(global_board_x, global_board_o, board_idx):
        return False
    # check that the move is in bounds
    # TODO: remove later, as ai can only return in bound moves
    if move < 0 or move > 80:
        return False
    # check that the move is not already taken
    if isSetOnBoard(board_x, board_o, move):
        return False

    return True
        
    

def getPossibleMoves(global_board_x, global_board_o, baord_x, board_o, currentPlayer, next_move_board_idx, winner):
    if winner is not Status.RUNNING:
        return []
    
    possible_moves = []

    if next_move_board_idx == 9:
        move_mask = 1
        board_mask = baord_x | board_o
        for block in range(9):
            if isNotPlayableBoard(global_board_x, global_board_o, block):
                continue

            for i in range(9):
                if not (board_mask & (move_mask << i)):
                    possible_moves.append(i + block * 9)
    else:
        if isNotPlayableBoard(global_board_x, global_board_o, next_move_board_idx):
            return []
        
        move_mask = 1 << (next_move_board_idx * 9)
        board_mask = baord_x | board_o
        for i in range(9):
            if not (board_mask & (move_mask << i)):
                possible_moves.append(i + next_move_board_idx * 9)

    return possible_moves


#---------------------------------
def apply_symmetry(bits: int, perm: list[int]) -> int:
    result = 0
    for i, j in enumerate(perm):
        if (bits >> j) & 1:
            result |= (1 << i)
    return result

def generate_all_symmetries(bits: int, map) -> dict[str, int]:
    return [apply_symmetry(bits, perm) for perm in map]

def get_symmetries(global_state_x, global_state_o, local_state_x, local_state_o, currentPlayer, currentBoard, winner):
    symmetries = []
    

    global_symmetries_x = generate_all_symmetries(global_state_x, SYMMETRY_INDICES) #len = 8
    global_symmetries_o = generate_all_symmetries(global_state_o, SYMMETRY_INDICES)

    local_symmetries_x = [generate_all_symmetries(board, LOCAL_SYMMETRY_INDICES) for board in local_state_x]
    local_symmetries_o = [generate_all_symmetries(board, LOCAL_SYMMETRY_INDICES) for board in local_state_o]

    print(local_symmetries_x)

    #for each symmetry
    for i in range(8):
        new_global_state_x = global_symmetries_x[i]
        new_global_state_o = global_symmetries_o[i]
        new_local_state_x = [0] * 9
        new_local_state_o = [0] * 9

        #for each local board
        for j in range(9):
            print(f"sym no.: {i}, small square: {j}, mapping to: {SYMMETRY_INDICES[i][j]}")
            new_local_state_x[SYMMETRY_INDICES[i][j]] = local_symmetries_x[j][i]
            new_local_state_o[SYMMETRY_INDICES[i][j]] = local_symmetries_o[j][i]
            # new_local_state_o[j] = local_symmetries_o[i][SYMMETRY_INDICES[i][j]]

        new_currentBoard = 9 if 9 == currentBoard else SYMMETRY_INDICES[i][currentBoard]
        symmetries.append((new_global_state_x, new_global_state_o, new_local_state_x, new_local_state_o, currentPlayer, new_currentBoard, winner))

    return symmetries
#---------------------------------



def stringRep(global_baord_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner):
    # returns a string representation of the board
    # 0 is empty, 1 is x, 2 is o
    board = ""
    for i in range(3):
        for j in range(3):
            if (global_baord_x & global_board_o) & (1 << (i * 3 + j)):
                board += "D"
            elif global_baord_x & (1 << (i * 3 + j)):
                board += "X"
            elif global_board_o & (1 << (i * 3 + j)):
                board += "O"
            else:
                board += "."
            # print vertical line
            if j != 2:
                board += "|"

        board += "\n"
    board += "\n"

    #for each row
    for i in range(9):
        for j in range(9):
            board_idx = i // 3 * 3 + j // 3
            local_x = j % 3
            local_y = i % 3
            move_mask = 1 << (board_idx * 9 + local_y * 3 + local_x)
            if board_x & move_mask:
                board += "X"
            elif board_o & move_mask:
                board += "O"
            else:
                board += "."
            board += " "
            if (j % 3 == 2) and (j != 8):
                board += "|"
        # print horizontal line
        board += "\n"
        if i % 3 == 2 and i != 8:
            board += "- - - - - - - - - \n"

    board += "\n"
    return board
        

if __name__ == "__main__":
    print(checkWin(INITIAL_STATE[2], 0))
    

