import asyncio
import websockets
import json
import argparse
import selfPlayEngine
import GameRepresentationFunctional as GRF

import random

#{last_move, player, board_state}

class TicTacToeClient:
    def __init__(self, uri):
        self.uri = uri
        self.websocket = None

    # Connect to the server
    async def connect(self):
        self.websocket = await websockets.connect(self.uri, ping_interval=None, ping_timeout=None)
        print(f"Connected to the server at {self.uri}")

    # Disconnect from the server
    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from the server.")

    # Receive a message from the server (can be last_move or opponent's move)
    async def receive_move_message_or_none(self):
        msg = await self.websocket.recv()
        try:
            server_msg = json.loads(msg)
            print("Received JSON from server:", server_msg)
            return server_msg
        except json.JSONDecodeError:
            print("Received non-JSON message:", msg)
            return None  # or handle this case as needed

    # Send a move to the server
    async def make_move(self, move):
        move_msg = {"move": list(move)}
        await self.websocket.send(json.dumps(move_msg))
        print("Sent to server:", move_msg)

    def parse_server_msg(self, last_move, player, state):
        #setting local boards
        local_state_x = [0] * 9
        local_state_o = [0] * 9
        for i in range(len(state)):
            local_state_x[i] = sum((1 << i) for i, val in enumerate(state[i]) if val == 1)
            local_state_o[i] = sum((1 << i) for i, val in enumerate(state[i]) if val == 2)

        #setting global state
        global_state_x = 0
        global_state_o = 0

        for i in range(9):
            if GRF.checkWin(local_state_x[i]):
                global_state_x |= (1 << i)
            if GRF.checkWin(local_state_o[i]):
                global_state_x |= (1 << i)
        
        #setting currentPlayer
        currentPlayer = "X" if player == 1 else "O"

        #setting currentBoard
        currentBoard = 9 if last_move == None else last_move[0] % 3 + last_move[1] // 3 * 3
        if GRF.isNotPlayableBoard(global_state_x, global_state_o, currentBoard):
            currentBoard = 9

        # setting winner
        winner = None
        game_state = (global_state_x, global_state_o, local_state_x, local_state_o, currentPlayer, currentBoard, winner)
        print(game_state)
        return game_state

async def main(uri):
    client = TicTacToeClient(uri)

    # Connect to the server
    await client.connect()

    while True:
        # Get initial last_move or game state
        server_msg = await client.receive_move_message_or_none()
        if(server_msg is None):
            return None  # handle "Winner X" => restart model

        last_move = server_msg.get("last_move", None)
        state = server_msg.get("game_state", None)
        player = server_msg.get("player", None)


        if(last_move == None or state == None):
            print("No message from server...")

        print(last_move) #debugging
        print(state) # debugging

        # state = (global_state x, global state o, local state x, local state o, currentplayer, currentboard, winner)
        game_state = client.parse_server_msg(last_move,player,state)
        

        # calc move
        my_move = selfPlayEngine.best_move(*game_state)  

        # Make a move
        await client.make_move(my_move)

    # Disconnect from the server
    await client.disconnect()



def test():
    client = TicTacToeClient("idk")
    arr = arr = [
    [0, 2, 1, 0, 0, 1, 2, 0, 0],
    [1, 0, 0, 2, 1, 0, 0, 2, 1],
    [2, 1, 0, 0, 2, 1, 0, 0, 2],
    [0, 0, 1, 2, 0, 0, 1, 2, 0],
    [1, 2, 0, 0, 1, 2, 0, 0, 1],
    [0, 1, 2, 0, 0, 1, 2, 0, 0],
    [2, 0, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 2, 0, 0, 1, 2, 0, 0],
    [1, 2, 0, 0, 1, 2, 0, 0, 1]
]
    res = client.parse_server_msg([0,0],2, arr)
    print(GRF.stringRep(*res))


if __name__ == "__main__":
    test()
'''

    # Argument parsing to allow passing WebSocket URI from command line
    parser = argparse.ArgumentParser(description="Connect to the Tic-Tac-Toe WebSocket server.")
    parser.add_argument("uri", help="WebSocket server URI (e.g., ws://localhost:PORT)")

    args = parser.parse_args()

    # Run the client with the provided WebSocket URI
    asyncio.run(main(args.uri))
'''
