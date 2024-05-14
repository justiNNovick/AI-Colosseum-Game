# Student agent: Add your own agent here
'''
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
        '''
#add agent here
from tracemalloc import start
from types import NoneType
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

 
import random
import itertools
import math
from collections import deque


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def random_wall(self, chess_board, pos):
        r, c = pos
        #get a random wall in any position
        fin = []
        for i in range(4):
            if not chess_board[r][c][i]:
                fin.append(i)
        return fin[random.randint(0, len(fin)-1)]
        
    def number_of_walls(self, chess_board, my_pos):
        r, c = my_pos
        numwall = 0
        for bol in chess_board[r, c]:
            if bol == True:
                numwall = numwall + 1
        return numwall

    def movesleft(self, chess_board, my_pos, adv_pos, max_step):
        r, c = my_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        #must keep moves as defined aboved to keep track of is possible
        possible_moves = set()
        q = deque([((r, c), 0)])
        visited = set()
        #run BFS to get possible moves
        while q:
            current, steps_taken = q.popleft()
            cur_x, cur_y = current[0], current[1]
            if steps_taken >= max_step:
                #if we reach the end then break
                break
            for m in moves:
                xlen, ylen = m
                new_position = ((cur_x + xlen), (cur_y + ylen))
                #check here to save some time
                if new_position in visited:
                    continue
                visited.add(new_position)
                #see if there are no obsatcles in the way and can move to next position
                if self.is_valid_pos(new_position, adv_pos, chess_board) and not chess_board[cur_x, cur_y, moves.index(m)]:
                    q.append(((new_position), steps_taken + 1))
                    possible_moves.add(new_position)
        possible_moves_remaining = len(possible_moves)
        #return number of moves remaining and walls
        return possible_moves_remaining, self.number_of_walls(chess_board, my_pos)
    #return n
        
    
    def successor_generate(self, chess_board, my_pos, adv_pos, max_step):
        r, c = my_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        possible_moves = set()
        q = [(r, c, 0)]  # Queue with elements as (x, y, steps_taken)
        visited = set()
        while q:
            cur_x, cur_y, steps_taken = q.pop(0)  # Using list as a queue
            #if we reached the end then break
            if steps_taken >= max_step:
                break
            for m in moves:
                new_x, new_y = cur_x + m[0], cur_y + m[1]
                new_position = (new_x, new_y)
                #check here to save time
                if new_position in visited:
                    continue
                visited.add(new_position)
                #if we can move to the next place then do it
                if self.is_valid_pos(new_position, adv_pos, chess_board) and not chess_board[cur_x][cur_y][moves.index(m)]:
                    q.append((new_x, new_y, steps_taken + 1))
                    possible_moves.add(new_position)
        #sorts our move by our utility heuristic to get the best possible move first
        sorted_possible_moves = sorted(possible_moves, key = lambda move: self.calculate_utility(chess_board, move, adv_pos, max_step), reverse=True)
        #return sorted moves
        return sorted_possible_moves
    
    def is_valid_pos(self, pos, opp_pos, chess_board):
        #make sure we are moving to a valid position
        x, y = pos
        xopp, yopp = opp_pos
        if x in range(0, len(chess_board)) and y in range(0, len(chess_board)) and (x != xopp or y!= yopp):
            return True
        else:
            return False

    
    def is_terminal_with_score(self, chess_board, my_pos, adv_pos):
        #taken from your world.py file since you said there is good code to look for over there
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        moves123 = ((-1, 0), (0, 1), (1, 0), (0, -1))
        board_size = len(chess_board)
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves123[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        if player_win == 0:
            return True, p0_score
        elif player_win == 1:
            return True, -190
        else:
            return True, 0
        
    def wall_heuristic(self, chess_board, my_pos, adv_pos, max_steps):
        r,c = my_pos
        possible_walls = []
        for i in range(4):
            if chess_board[r][c][i] == False:
                possible_walls.append(i)
        maxval = float('-inf')
        best_wall = possible_walls[0]
        for wall in possible_walls:
            changes = self.simulate_move_for_isTerminal(chess_board, my_pos, wall)
            score = self.calculate_utility(chess_board, my_pos, adv_pos, max_steps)
            self.undo_changes(chess_board, changes)
            if score > maxval:
                best_wall = wall
                maxval = score
        return best_wall
    
    def calculate_utility(self, chess_board, my_pos, adv_pos, max_step):
        my_moves_left, my_walls = self.movesleft(chess_board, my_pos, adv_pos, max_step)
        opp_moves_left, opp_walls = self.movesleft(chess_board, adv_pos, my_pos, max_step)

        #if opponent is trapped in then make utility very good and play aggressive
        if opp_moves_left < 5:
            return 10000
        else:
            #make utility a combination of how many moves left and how many walls surround each other
            score = (1.2 * my_moves_left - opp_moves_left) + pow(3, opp_walls) - pow(3, my_walls)
            return score

    def simulate_move(self, chess_board, new_pos, opp_pos, max_step, wall): 
        board_copy = deepcopy(chess_board)
        r, c = new_pos
        board_copy[r,c,wall] = True
        
        # Determine the corresponding wall on the adjacent cell
        if wall == 0:
            otherWall = 2
            otherPos = (r-1, c)
        elif wall == 1:
            otherWall = 3
            otherPos = (r, c+1)
        elif wall == 2:
            otherWall = 0
            otherPos = (r+1, c)
        else:  # wall == 3
            otherWall = 1
            otherPos = (r, c-1)

        otherR, otherC = otherPos
        if otherR in range(len(board_copy)) and otherC in range(len(board_copy)):
            board_copy[otherR, otherC, otherWall] = True
        return board_copy

    def simulate_move_for_isTerminal(self, chess_board, new_pos, wall): 
        r, c = new_pos
        changes = []
        chess_board[r,c,wall] = True
        changes.append((r, c, wall))
        
        if wall == 0:
            otherWall = wall + 2
            otherPos = (r-1, c)
        elif wall == 1:
            otherWall = wall + 2
            otherPos = (r, c+1)
        elif wall == 2:
            otherWall = wall - 2
            otherPos = (r+1, c)
        else:    #wall == 3
            otherWall = wall -2
            otherPos = (r, c-1)
        #only putting wall if both are valid.... we only care about other....
        
        otherR, otherC = otherPos
        if otherR in range(0, len(chess_board)) and otherC in range(0, len(chess_board)):
            chess_board[otherR, otherC, otherWall] = True
            changes.append((otherR, otherC, otherWall))
        return changes
    

    def undo_changes(self, chess_board, changes_lst):
        if len(changes_lst) == 0:
            return
        for r, c, wall in changes_lst:
            chess_board[r,c,wall] = False
        #revert board bake to normal
        
    def is_terminal_minimax(self, chess_board, my_pos, adv_pos):
        #easier version of is_terminal that takes less time
        r,c = my_pos
        x, y = adv_pos
        possible_barriers = [i for i in range(4) if not chess_board[r, c,i]]
        adv_barriers = [i for i in range(4) if not chess_board[x,y,i]]
        if len(possible_barriers) == 0 or len(adv_barriers) == 0:
            return True
        else:
            return False
    
        
    
    def minimax(self, depth, chess_board, my_pos, adv_pos, alpha, beta, maximizingPlayer, max_step, time_limit, start_time, is_first_call):
        current_time = time.time()
        #this is our base case to see if we ran out of time or is_terminal or hit the depth
        if current_time - start_time > time_limit or depth == 0 or self.is_terminal_minimax(chess_board, my_pos, adv_pos):
            return self.calculate_utility(chess_board, my_pos, adv_pos, max_step), None, None

        if maximizingPlayer:
            maxEval = float('-inf')
            best_move = None
            best_wall = None
            #loop over all moves sorted by utility
            for move in self.successor_generate(chess_board, my_pos, adv_pos, max_step):
                #only check for every wall the first call every other call dont check for walls
                walls_to_try = range(4) if is_first_call else [random.choice([i for i in range(4) if not chess_board[move[0]][move[1]][i]])]
                for wall in walls_to_try:
                    r, c = move[0], move[1]
                    if not chess_board[r][c][wall]:
                        board_copy = self.simulate_move(chess_board, move, adv_pos, max_step, wall)
                        #run minimax and get score back
                        eval, _, _ = self.minimax(depth - 1, board_copy, move, adv_pos, alpha, beta, False, max_step, time_limit, start_time, False)
                        if eval > maxEval:
                            maxEval, best_move, best_wall = eval, move, wall
                        #update our alpha value
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return maxEval, best_move, best_wall
        else:
            minEval = float('inf')
            best_move = None
            best_wall = None
            for move in self.successor_generate(chess_board, my_pos, adv_pos, max_step):
                #run minimax for min player
                #only check for every wall the first call every other call dont check for walls
                walls_to_try = range(4) if is_first_call else [random.choice([i for i in range(4) if not chess_board[move[0]][move[1]][i]])]
                for wall in walls_to_try:
                    #if we can move to this place then:
                    if not chess_board[move[0]][move[1]][wall]:
                        board_copy = self.simulate_move(chess_board, my_pos, move, max_step, wall)
                        #generates hard copy
                        #run minimax and get score back as hueristic 
                        eval, _, _ = self.minimax(depth - 1, board_copy,  my_pos, adv_pos, alpha, beta, True, max_step, time_limit, start_time, False)
                        if eval < minEval:
                            minEval, best_move, best_wall = eval, move, wall
                        #update the beta value that we have
                        beta = min(beta, eval)
                        if alpha >= beta:
                            #prune
                            break
            return minEval, best_move, best_wall
   

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.num_walls = 2
        

    def step(self, chess_board, my_pos, adv_pos, max_step):
        orig_pos = deepcopy(my_pos)
        start_time = time.time()
        best_move = None
        best_wall = None
        best_utility = float('-inf')
        if max_step > 4:
            d = 2
        else:
            d = 3

        # Iterate through each possible move
        for index, move in enumerate(self.successor_generate(chess_board, my_pos, adv_pos, max_step)):
            r, c = move
            losing_var = False

            #if time exceeded
            if time.time() - start_time >= 1.6:
                break

            for i in range(4):
                #simulate every possible wall placement
                if chess_board[r][c][i] == False:
                    #see if we can win right away
                    #changes = self.simulate_move_for_isTerminal(chess_board, move, i)
                    self.simulate_move(chess_board, move, adv_pos, max_step, i)
                    bol, score = self.is_terminal_with_score(chess_board, move, adv_pos)
                    #if winning move right away then go straigh to winning move
                    if bol and score > 0:
                        return move, i
                    elif bol and score < 0:
                        losing_var = True

                    #self.undo_changes(chess_board, changes)

                if losing_var:
                    continue
        start_time = time.time()
        #Run minimax after we see that we cant win right away
        utility, best_move, best_wall = self.minimax(depth=d, chess_board=chess_board, my_pos=my_pos, adv_pos=adv_pos, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=True, max_step=max_step, time_limit=1.58, start_time=start_time, is_first_call=True)
        # After iterating through all moves or if time limit is exceeded
        if best_move is not None and best_wall is not None:
            return best_move, best_wall
        else:
            return my_pos, best_wall
