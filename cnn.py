# max_pooling on conv
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 342 will be 2 4 3
        # 465 will be 5 6 4
        while (True):
            if l1 == None or l2 == None:
                pass



# def main():
#     l1 = ListNode(2)
#     l1.next = ListNode(4)
#     l1.next.next = ListNode(3)
#
#     l2 = ListNode(5)
#     l2.next = ListNode(6)
#     l2.next.next = ListNode(4)
#
#     sol = Solution()
#     obj = sol.addTwoNumbers(l1, l2)
########################################################################
# graphics

# user input co-ordinate,
# occupied_cell =
# print("Pick your cell\n")
#
# print ()
# player1_input = int(input())
# 1

print ("Welcome to Aloo Cross")

board = [[]]

board = [[i for i in range(3)] for j in range(3)]

c = 1
coordinates = {}

for i in range(3):
    for j in range(3):
        board[i][j] = c
        coordinates[c] = (i,j)
        c += 1

def printBoard(board):
    scale = 1
    for j in board:
        for i in j:
            if len(str(i)) == 2:
                print(str(i) + ' |',end=" ")
            else:
                print(str(i) + '  |',end=" ")


        print("\n-------------")

printBoard(board)

### inputs ###
print("Player 1 name:\n",)
p1 = input()
print("Player 2 name:\n",)
p2 = input()

print("{} is player 1, {} is player 2".format(p1, p2))

name = {1: p1+" (player1)", 2: p2 + " (player2)"}
print(coordinates)

#def cell_selection(i):

def check_finish(player):
    # width, height and main diagnol
    # returns 1 if finished, 0 if more remains to be played
    # horizontal check

    for i in range(3):
        star_freq = 0
        o_freq = 0
        for j in range(3):
            if board[i][j] == '*':
                star_freq  += 1
            elif board[i][j] == 'o':
                o_freq += 1
        if star_freq == 3 or o_freq == 3:
            print("{0} wins. Congratulations! Winner winner biryani dinner Game ended".format(name[player]))
            return True

    #vertical check
    for i in range(3):
        star_freq = 0
        o_freq = 0
        for j in range(3):
            if board[j][i] == '*':
                star_freq  += 1
            elif board[j][i] == 'o':
                o_freq += 1
        if star_freq == 3 or o_freq == 3:
            print("{0} wins. Congratulations! Winner winner biryani dinner Game ended".format(name[player]))
            return True

    # diagnol check
    # dg 1
    if board[0][0] == board[1][1] == board[2][2] == "*":
        print("{0} wins. Congratulations! Winner winner biryani dinner Game ended".format(name[1]))
        return True
    elif board[0][0] == board[1][1] == board[2][2] == "o":
        print("{0} wins. Congratulations! Winner winner biryani dinner Game ended".format(name[2]))
        return True
    #dg 2
    if board[0][2] == board[1][1] == board[2][0] == "*":
        print("{0} wins. Congratulations! Winner winner biryani dinner Game ended".format(name[1]))
        return True
    if board[0][2] == board[1][1] == board[2][0] == "o":
        print("{0} wins. Congratulations! Winner winner biryani dinner Game ended".format(name[2]))
        return True

    return False

game_finish = False
winner = -1
printBoard(board)
while not game_finish:
    print("{} 's turn to choose a cell:".format(name[1]))
    p1 = int(input())
    if p1 in coordinates:
        # p1_coordinate = (_,_)
        p1_coordinate = coordinates[p1]
        board[p1_coordinate[0]][p1_coordinate[1]] = '*'
        coordinates.pop(p1)
    else:
        print("Enter valid cell number")
        continue
    printBoard(board)
    game_finish = check_finish(1)
    if len(coordinates) <= 3:
        print("Game drawn. No biryani for both")
        break
    if game_finish:
        break
    print("{} 's turn to choose a cell:".format(name[2]))
    valid_input_p2 = False
    # no while loop for p1 as can just continue
    while not valid_input_p2:
        p2 = int(input())
        if p2 in coordinates:
            # p1_coordinate = (_,_)
            valid_input_p2 = True
            p2_coordinate = coordinates[p2]
            board[p2_coordinate[0]][p2_coordinate[1]] = 'o'
            coordinates.pop(p2)
        else:
            print("Enter valid cell number")
    printBoard(board)
    game_finish = check_finish(2)
    if len(coordinates) <= 3:
        print("Game drawn. No biryani for both")
        break
#  tester
