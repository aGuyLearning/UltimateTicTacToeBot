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

    

INITIAL_STATE = (
    0b0, #global_state x
    0b0, # global state o
    0b0, # local state x
    0b0, # local state o
    True, # currentplayer
    9, # 0-8 are the local boards, 9 is free choice
    None
)

ZOBRIST_TABLE = {
    "squares": [
            [3903901466010275279, 17629543429219029125],
            [6560537333460717463, 7986740484246722560],
            [13448444757883081222, 8862075095357973495],
            [627530935268485925, 13618924972420855844],
            [5658841495040586966, 10588138687755045676],
            [13375152968863883301, 11412907888164986633],
            [3720144703103354770, 754926789191194911],
            [10153693656287081050, 7904381750307070847],
            [5989228250479128887, 13415111613617292659],
            [12059086116899244332, 7451163809502394975],
            [14996391777122309193, 10143908630119605033],
            [10754802737566917926, 1088440516914160163],
            [8386103541441791909, 12542817629116669989],
            [238424499285833975, 9653130381207852254],
            [9405888958724483071, 17126726561198328420],
            [8179610268055778624, 6553341512026093135],
            [15560361405764370159, 16746408262836215598],
            [2644246315876859064, 2061237395763558784],
            [16124045022113393060, 14256613376501550053],
            [14021379837533551897, 6802895348214381347],
            [136544599174301899, 9168882058527334167],
            [3011668097940964029, 10773388103513099196],
            [4335971041406852442, 13718617498506983821],
            [9218116851555331607, 10002078858339948169],
            [3740883737439103451, 2027739035571163026],
            [4671695329495669244, 10164495824027210026],
            [2930388333517988396, 11791177830834100646],
            [12195637642012800609, 4818057183511173711],
            [5746580025108558176, 8593898498462459111],
            [6915933671817634910, 500010199411656409],
            [17160765776457446381, 17499166974477630742],
            [7531030024674436671, 14444020855268728990],
            [14420961019011337767, 9953309611959569954],
            [12755489774125999224, 13386131398488311337],
            [10572301849303105890, 14245356919393356844],
            [7749810252646487439, 405107600915105234],
            [4673941769004790044, 9912376699992536506],
            [13772861075326472726, 1424635966634839591],
            [10448224672211504257, 10375061792833773117],
            [810972899958790168, 297139134030571911],
            [3433409098337369845, 4634971037386423006],
            [12850522352033757384, 6596947357672707500],
            [13778958088203008837, 14423899773881843734],
            [8812853012364468235, 14203791236973137482],
            [4374649817425260023, 2753676664979010312],
            [986597567356751249, 15570697413009087723],
            [15530147278100853657, 17421029532736007804],
            [2527105967667774100, 11286137556684937497],
            [304139287659296867, 7540826990568408152],
            [10166453674469440684, 6099008868128817969],
            [6622564022921803344, 9408288026298252096],
            [8349288625820524788, 2373305270142202147],
            [17564047435738048593, 234641463252374735],
            [15764970752357614869, 11458980784115948641],
            [12818556854687771997, 156920060851728499],
            [18274899741850289430, 9654313182984924307],
            [12255631505909503829, 865491575069738653],
            [17256558342343098752, 17955956082692613109],
            [578210944565411855, 3600222308699921328],
            [3031820988293496544, 302500152432294006],
            [2747199606362250167, 9452470642626238316],
            [7808060539223555787, 15468571447339663953],
            [11158004304530322043, 16263219473716500958],
            [17244338066467202728, 9053834545123193093],
            [11453274589915392600, 6495049422195869991],
            [6318660622973812624, 7950284959269046255],
            [1293341356545473800, 6354918147291769233],
            [12027263893405586912, 2827004469857518461],
            [2510899281429553955, 5677839733827651068],
            [15477038650223642817, 3714445228525652981],
            [14711424822796252471, 645043225026648688],
            [14451820229844848118, 16433456985459467283],
            [10734409564733582363, 17888608904115112824],
            [14963165677727213545, 13735647310550505811],
            [10977874651871842400, 12388199409902683965],
            [18402500573870520482, 17645806566650012533],
            [7617244385618196477, 2403187083095025443],
            [15534413895419439910, 15322806809198407270],
            [6998107393372259430, 17484518793461397802],
            [1805771003956062686, 15508142961429924035],
            [14983629085325617298, 13988792144298142757],
    ],
    "boards": [
        [
            10569293916965141230,
            11160423188805547020,
            4152050228531721776,
        ],
        [
            9445060042156387674,
            17343938531420928522,
            4567518214753125608,
        ],
        [
            10044836178330282261,
            6985788723661997541,
            2957325009523696543,
        ],
        [
            4262220721745283853,
            9629117720159326359,
            14144349062870192621,
        ],
        [
            15701379455324056445,
            13805334679043656525,
            2244893216642584252,
        ],
        [
            14647228938910723007,
            1431351858910568268,
            1852814477193620178,
        ],
        [
            10516372127856826326,
            12162076319882157197,
            4528292767345948788,
        ],
        [
            8820780412613772490,
            11456186954002538683,
            7988953696186528238,
        ],
        [
            11349973509132415302,
            8414190152471347161,
            14547444142552861968,
        ],
    ],
    "next_board": [
        16798470787979032810,
        13172679987797492821,
        8295043729246619448,
        471941106522091992,
        16298057341796429847,
        7135075252472040655,
        9733518850052539775,
        5430438456607159556,
        7693123149093558032,
    ],
    "players": [5407203028010692663, 14035603769899813055],
}
def zobrist(global_state_x, global_state_o, board_x, board_o, currentPlayer, currentBoard, winner):
    h = 0
    
    # Hash the local boards

    for pos in range(81):
        if board_x & (1 << pos):
            h ^= ZOBRIST_TABLE["squares"][pos][0]
    
    # Hash O's pieces on this local board
    for pos in range(81):
        if board_o & (1 << pos):
            h ^= ZOBRIST_TABLE["squares"][pos][1]
    

    # Hash the global boards (won boards)
    for board_idx in range(9):
        if global_state_x & (1 << board_idx):
            h ^= ZOBRIST_TABLE["boards"][board_idx][0]
        if global_state_o & (1 << board_idx):
            h ^= ZOBRIST_TABLE["boards"][board_idx][1]
        # For drawn boards (both X and O have the bit set)
        if (global_state_x & global_state_o) & (1 << board_idx):
            h ^= ZOBRIST_TABLE["boards"][board_idx][2]
    
    # Hash the current board
    if currentBoard != 9:  # 9 means free choice
        h ^= ZOBRIST_TABLE["next_board"][currentBoard]
    
    # Hash the current player
    h ^= ZOBRIST_TABLE["players"][0 if currentPlayer else 1]
    
    return h






def move(global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner, move):
        
    if (checkValidMove(global_board_x, global_board_o, board_x, board_o, next_move_board_idx, move)):
        board_idx = move // 9

        if currentPlayer:
            board_x |= (1 << move)
            if checkWin(board_x, board_idx):
                global_board_x |= (1 << board_idx)
                if checkWin(global_board_x & ~global_board_o):
                    winner = 1
                    return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)                   
                elif checkDraw(global_board_x, global_board_o):
                    winner = 0
                    return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner) 
        else:
            board_o |= (1 << move)
            if checkWin(board_o, board_idx):
                global_board_o |= (1 << board_idx)
                if checkWin(global_board_o & ~global_board_x):
                    winner = -1
                    return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)
                elif checkDraw(global_board_x, global_board_o):
                    winner = 0
                    return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)
                
        if checkDraw(board_x, board_o, board_idx):
            global_board_x |= (1 << board_idx)
            global_board_o |= (1 << board_idx)
            if checkDraw(global_board_x, global_board_o):
                winner = 0
                return (global_board_x, global_board_o, board_x, board_o, currentPlayer, next_move_board_idx, winner)
            
        next_move_board_idx = move % 9
        if isNotPlayableBoard(global_board_x, global_board_o, next_move_board_idx):
            next_move_board_idx = 9
        
        return (global_board_x, global_board_o, board_x, board_o, not currentPlayer, next_move_board_idx, winner)
    return None
        
### local functions:
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
    if winner is not None:
        return []
    
    possible_moves = []

    if next_move_board_idx == 9:
        move_mask = 1
        board_mask = baord_x | board_o
        for block in range(9):
            if isNotPlayableBoard(global_board_x, global_board_o, block):
                continue

            for i in range(9):
                if not (board_mask & (move_mask << (i + block * 9))):
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


    #for each symmetry
    for i in range(8):
        new_global_state_x = global_symmetries_x[i]
        new_global_state_o = global_symmetries_o[i]
        new_local_state_x = [0] * 9
        new_local_state_o = [0] * 9

        #for each local board
        for j in range(9):
            new_local_state_x[SYMMETRY_INDICES[i][j]] = local_symmetries_x[j][i]
            new_local_state_o[SYMMETRY_INDICES[i][j]] = local_symmetries_o[j][i]
            # new_local_state_o[j] = local_symmetries_o[i][SYMMETRY_INDICES[i][j]]

        new_currentBoard = 9 if 9 == currentBoard else SYMMETRY_INDICES[i][currentBoard]
        symmetries.append((new_global_state_x, new_global_state_o, new_local_state_x, new_local_state_o, currentPlayer, new_currentBoard, winner))

    return symmetries

def flip_arr(arr):
    # inner flipping
    internal_flipped = []
    for perm in range(len(SYMMETRY_INDICES)):
        internal_rows = []
        for row in range(len(arr)//9):
        for row in range(len(arr)//9):
            new_row = [0] * 9
            for i in range(9):
                new_row[SYMMETRY_INDICES[perm][i]] = arr[9*row + i]
                new_row[SYMMETRY_INDICES[perm][i]] = arr[9*row + i]
            internal_rows.append(new_row)
        internal_flipped.append(internal_rows)
    # outer flipping
    res = []
    res = []
    new_arr = [[0] * 9 for _ in range(8)]
    for perm in range(len(SYMMETRY_INDICES)):
        for i in range(9):
            new_arr[perm][SYMMETRY_INDICES[perm][i]] = internal_flipped[perm][i]
        res.append(sum(new_arr[perm],[]))
    return res
        

        


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
    arr = [
        1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    symmetries_result = flip_arr(arr)
    for perm in range(8):
            print(symmetries_result[perm])
            print()

    '''
    # set start time 
    start = time.time()
    for _ in range(10**5):
        game = copy.deepcopy(INITIAL_STATE)
        for i in range(100):
            possible_moves = getPossibleMoves(*game)
            if len(possible_moves) == 0:
                # print(f"Game over after {i} moves")
                break
            my_move = random.choice(possible_moves)
            game = move(*game, my_move[0], my_move[1])
    
    end = time.time()

    print(f"Time taken: {end - start} seconds")
    
    '''
