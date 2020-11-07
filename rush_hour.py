import sys
from queue import PriorityQueue
import copy
import time

def list_to_string(arr): #convert list to string
    str_node=''
    for i in range(0, Puzzle.size_table):
        for j in range(0, Puzzle.size_table):
            str_node += arr[i][j]
    return str_node
def print_array(ar): #print the array in the form of matrix
    print('\n'.join([''.join(['{:5}'.format(item) for item in row])
                     for row in ar]))

class Node:
    # constructor construct the node
    def __init__(self,state_of_puzzle,parent,depth,array_puzle,herstic_val,move):
        self.state_of_puzzle=state_of_puzzle #dictionary that save to every car thw location of the car in the array(puzle)and the type of the car and the type of the move
        self.parent=parent #save pionter to the parent
        self.depth=depth # save the depth
        self.array_puzle=array_puzle #save the array
        self.herstic_val=herstic_val
        self.move=move #save the last move that by this move we arrive to this node

    def __lt__(self, other): #the priorty quiue use this function to compare between the nodes that found in the quiue
        """
        :param other: compare
        :return: 0
        """

        return 0

    def goal_state(self): #if the red car found in the end of the third row so the red car can exit and we arrive to the solution
        return self.array_puzle[2][5]=='X'

    def score(self,herstic_id): #compute the f value of the node in according to the algortim we use it to find the solution
        if herstic_id=='iterative_deepning' or herstic_id=='DFS':
            return -self.depth
        elif herstic_id=='BFS':
            return self.depth
        elif herstic_id==9:
            return self.herstic_val
        elif herstic_id==3:
            return self.herstic_val-self.depth
        return self.depth+self.herstic_val
    #function that take the car that we want to move it and take the direction and the amount of move
    def create_new_node(self, cur_car, sum, direction,herstic_id):
        new_array_board = copy.deepcopy(self.array_puzle) #create a new array to the new node
        move=[]
        move.append(self.array_puzle[cur_car.index[0]][cur_car.index[1]])
        move.append(direction)
        move.append(str(sum))    #save the last move
        new_dict_car = dict(self.state_of_puzzle) #create new dictionary
        index = cur_car.index
        new_location_car = self.array_puzle[index[0]][index[1]]
        if cur_car.type_of_move==0: #according to the direction we change the array and change the location of the car
            if direction=='L':
                for i in range(0,cur_car.type_of_car+2):
                    new_array_board[index[0]][index[1]-sum+i]= new_array_board[index[0]][index[1]+i]
                    new_array_board[index[0]][index[1] + i]='.'
                index=(index[0],index[1]-sum)
            elif direction=='R':
                for i in range(cur_car.type_of_car+1,-1,-1):
                    new_array_board[index[0]][index[1]+sum+i]= new_array_board[index[0]][index[1]+i]
                    new_array_board[index[0]][index[1] + i]='.'
                index = (index[0], index[1] + sum)

        else:
            if direction=='U':
                for i in range(0,cur_car.type_of_car+2):
                    new_array_board[index[0]-sum+i][index[1]]= new_array_board[index[0]+i][index[1]]
                    new_array_board[index[0]+i][index[1]]='.'
                index=(index[0]-sum,index[1])
            else:

                for i in range(cur_car.type_of_car+1,-1,-1):
                    new_array_board[index[0]+sum+i][index[1]]= new_array_board[index[0]+i][index[1]]
                    new_array_board[index[0]+i][index[1]]='.'
                index=(index[0]+sum,index[1])
        str_newarray=list_to_string(new_array_board)
        if herstic_id=='iterative_deepning':
            if str_newarray not in Iterative_deepning.visited:
                depth = self.depth + 1
                car = Car(index, cur_car.type_of_car, cur_car.type_of_move)
                new_dict_car[new_location_car] = car
                new_parent = self
                new_node = Node(new_dict_car, new_parent, depth, new_array_board, None, move)
                new_node.herstic_val=0
                Iterative_deepning.waiting_nodes.put((new_node.score('iterative_deepning'), new_node))

        else:
            #if we visit the new node in the past so No need to go through it again
            if str_newarray not in Puzzle.visited:
                depth = self.depth + 1
            #Puzzle.number_of_nodes += 1
                car = Car(index, cur_car.type_of_car, cur_car.type_of_move)
                new_dict_car[new_location_car] = car
                new_parent = self
                new_node = Node(new_dict_car, new_parent, depth, new_array_board, None, move)
                if herstic_id==4:
                    val_herstic = new_node.calc_herustic_valtemp()
                else:
                    val_herstic = new_node.calc_herustic_val(herstic_id)
                new_node.herstic_val = val_herstic
                Puzzle.add_to_waitlist(new_node,herstic_id)
    def how_to_move(self,cars_can_move): #function that take the cars the can move and compute to every car can move the direction of the move
        mov_car = {}
        for key in cars_can_move:
            curent_car = self.state_of_puzzle[key]
            location = curent_car.index
            count = curent_car.type_of_car + 2
            moves=(0,0)
            if curent_car.type_of_move == 0: #if the type of the move is horizontal
                if location[1] - 1 >= 0 and self.array_puzle[location[0]][location[1] - 1] == '.':
                    moves=(1,moves[1]) #the car can move left L1
                if location[1] + count< Puzzle.size_table and self.array_puzle[location[0]][location[1]+count]=='.':
                        moves=(moves[0],1) #the car can move right R1
                mov_car[key]=moves #save the move to this car in the dictionary (the key is the car )
            elif curent_car.type_of_move==1:#if the type of the move is vertical
                if location[0] - 1 >= 0 and self.array_puzle[location[0]-1][location[1]] == '.':
                    moves=(1,moves[1]) #the car can move up U1
                if location[0] + count < Puzzle.size_table and self.array_puzle[location[0]+count][location[1]]=='.':
                    moves=(moves[0],1)#the car can move down D1
                mov_car[key]=moves
        return mov_car #return the dictionary the key is the cars and the values is the moves




    def calc_successors(self,herstic_id):#compute the children of the node
        succesors = []
        for succ in self.state_of_puzzle:
            location = self.state_of_puzzle[succ].index
            suc_car = self.state_of_puzzle[succ]
            if suc_car.type_of_move == 0:              # if no find space near the car so the car can't move so we don't save it in the list
                if location[1] % 6 == 0:
                    if self.array_puzle[location[0]][location[1] + suc_car.type_of_car + 2] != '.':
                        continue
                elif (location[1] + suc_car.type_of_car + 1) % 6 == 5:
                    if self.array_puzle[location[0]][location[1] - 1] != '.':
                        continue
                elif self.array_puzle[location[0]][location[1] + suc_car.type_of_car + 2] != '.' and \
                        self.array_puzle[location[0]][location[1] - 1] != '.':
                    continue
                succesors.append(succ)
            else:
                if location[0] % 6==0:
                    if self.array_puzle[location[0]+suc_car.type_of_car+2][location[1]] != '.':
                        continue

                elif (location[0]+suc_car.type_of_car+1)%6==5:
                    if self.array_puzle[location[0]-1][location[1]]!='.':
                        continue
                elif self.array_puzle[location[0]+suc_car.type_of_car+2][location[1]] != '.' and self.array_puzle[location[0]-1][location[1]]!='.':
                    continue


                succesors.append(succ)

        mov_car = self.how_to_move(succesors)
        for key in succesors:
            curent_car = self.state_of_puzzle[key]
            if curent_car.type_of_move==0:
                if mov_car[key][0]>0:
                    self.create_new_node(curent_car, mov_car[key][0], 'L',herstic_id)
                if mov_car[key][1]>0:
                    self.create_new_node(curent_car, mov_car[key][1], 'R',herstic_id)

            elif curent_car.type_of_move==1:
                if mov_car[key][0]>0:
                    self.create_new_node(curent_car, mov_car[key][0], 'U',herstic_id)
                if mov_car[key][1]>0:
                    self.create_new_node(curent_car, mov_car[key][1], 'D',herstic_id)
    def find_blocking_cars(self,blocking_cars,diff_locationcar):#function that take the cars that block the red car the cars that found in the third row after the red car
        #and the function finds if there are cars that block the path of the other cars that block the path of the red car if yes we save them and return them
        for key in diff_locationcar:
            if key=='X':
                continue
            move=diff_locationcar[key]
            cur_car=self.state_of_puzzle[key]
            for i in range(cur_car.index[0],cur_car.index[0]-move[0]-1):
                if i<0:
                    break
                elif self.array_puzle[i][cur_car.index[1]] in blocking_cars or self.array_puzle[i][cur_car.index[1]]=='.':
                    continue
                else:
                    blocking_cars.append(self.array_puzle[i][cur_car.index[1]])

            for i in range(cur_car.index[0],cur_car.index[0]+cur_car.type_of_car+2+move[1]):
                if i>Puzzle.size_table:
                    break
                elif self.array_puzle[i][cur_car.index[1]] in blocking_cars or self.array_puzle[i][ cur_car.index[1]]=='.':
                    continue
                else:
                    blocking_cars.append(self.array_puzle[i][cur_car.index[1]])

        return blocking_cars








    def calc_herustic_val(self,herstic_id): #the function compute the herstic value of the node
        index = self.state_of_puzzle['X'].index #save the location of the red car
        if index[0] == 2 and index[1] == Puzzle.size_table - 2: #if the red car found in the end of the third row so the red car can exit and return 0
            return 0


        new_location = []
        diff_locationcar={}
        new_location.append('X') #we save the cars that block the path of the red car also the red car
        sum = Puzzle.size_table - 1 - index[1] - 1
        for i in range(index[1],Puzzle.size_table):
            move=(0,0)
            if self.array_puzle[index[0]][i] not in new_location and self.array_puzle[index[0]][i]!='.':
                if self.array_puzle[index[0]][i]!='X'and self.state_of_puzzle[self.array_puzle[index[0]][i]].type_of_move==0:
                    return sys.maxsize #faluire if there is car that move horizontal that found after the red car so the red car can't exit
                car=self.state_of_puzzle[self.array_puzle[index[0]][i]]
                diff = index[0] - car.index[0]
                num_down = car.type_of_car + 2 - abs(diff) - 1
                if num_down + 1 <= car.index[0] % 6: #check if the car the block the path of the red car can move up and open the path to the red car
                    move=(num_down+1,move[1])
                    diff_locationcar[self.array_puzle[index[0]][i]] = move
                num_up = car.type_of_car + 2 - num_down - 1
                if num_up + 1 + car.index[0] + car.type_of_car + 1 <= Puzzle.size_table - 1:#check if the car the block the path of the red car can move down and open the path to the red
                    move=(move[0],num_up+1)
                    diff_locationcar[self.array_puzle[index[0]][i]] = move

                if move[0]<=move[1] and move[0]!=0:
                    sum+=move[0]
                elif move[1]<move[0] and move[1]!=0:
                    sum+=move[1]

                new_location.append(self.array_puzle[index[0]][i])
        if herstic_id==1 or herstic_id==9:
            new_location=self.find_blocking_cars(new_location,diff_locationcar)
        if herstic_id==3:
            return sum
        return len(new_location)
class Iterative_deepning:#class that run the algorthim iterative deepning
    visited = {}
    waiting_nodes = PriorityQueue()
    def iterative_deepning(root):

        size_table = 6
        number_of_nodes=0
        sum_herustic=0

        start = time.time()
        for i in range(0,sys.maxsize):
            Iterative_deepning.waiting_nodes = PriorityQueue()
            Iterative_deepning.visited={}
            Iterative_deepning.waiting_nodes.put((root.score('iterative_deepning'), root))
            while not Iterative_deepning.waiting_nodes.empty():
                cur_node = Iterative_deepning.waiting_nodes.get()[1]
                str_node = list_to_string(cur_node.array_puzle)
                while str_node in Iterative_deepning.visited and not Iterative_deepning.waiting_nodes.empty():
                    cur_node = Iterative_deepning.waiting_nodes.get()[1]
                    str_node = list_to_string(cur_node.array_puzle)

                if cur_node.depth>i:
                    continue
                Iterative_deepning.visited[str_node] = 1
                number_of_nodes += 1
                sum_herustic += cur_node.score('iterative_deepning')
                if time.time() - start>80:
                    print('failed')
                    return number_of_nodes,herstic_id,0,sum_herustic/number_of_nodes,number_of_nodes**(1/cur_node.depth),cur_node.depth,cur_node.depth,cur_node.depth,80
                elif cur_node.goal_state():
                    timeExe = time.time() - start

                    print("Time of execution:", timeExe)
                    if timeExe <= 80:
                        temp_node = cur_node
                        last_move = []
                        last_move.append('X')
                        last_move.append('R')
                        a = 2

                        last_move.append(str(a))
                        list_moves = []
                        count = 1

                        while temp_node.parent != None:

                            str_move = temp_node.move
                            if str_move[0] == last_move[0] and str_move[1] == last_move[1]:
                                count += 1
                            else:
                                str1 = last_move[0] + last_move[1] + str(count)
                                list_moves.append(str1)
                                count = 1
                                last_move = str_move
                            temp_node = temp_node.parent

                        if last_move != None:
                            str1 = last_move[0] + last_move[1] + str(count)
                            list_moves.append(str1)
                        temp_node = cur_node
                        max_depth = None
                        min_depth = None
                        size = 0
                        sum_depth = 0
                        while not Iterative_deepning.waiting_nodes.empty():
                            temp_node = Iterative_deepning.waiting_nodes.get()[1]
                            size += 1
                            sum_depth += temp_node.depth
                            if max_depth == None or max_depth < temp_node.depth:
                                max_depth = temp_node.depth
                            if min_depth == None or min_depth > temp_node.depth:
                                min_depth = temp_node.depth

                        avg_depth = 0
                        if size == 0:
                            avg_depth = 0
                            min_depth=cur_node.depth
                            max_depth=cur_node.depth
                        else:
                            avg_depth = sum_depth / size


                        count = len(list_moves)
                        for i in range(count - 1, -1, -1):
                            print(list_moves[i], end=" ")

                        print('\n number of visited nodes:', number_of_nodes)
                        print(' Penetrance: ', cur_node.depth / number_of_nodes)
                        print('avg H value: ', sum_herustic / number_of_nodes)
                        print('EBF: ', number_of_nodes ** (1 / cur_node.depth))
                        print('Max depth: ', max_depth)
                        print('Min depth: ', min_depth)
                        print('Average depth: ', avg_depth)
                        return number_of_nodes,'iterative_deepning',cur_node.depth/number_of_nodes,sum_herustic/number_of_nodes,number_of_nodes ** (1 / cur_node.depth),max_depth,min_depth,avg_depth,timeExe
                    else:
                        print('failed')

                else:
                    cur_node.calc_successors('iterative_deepning')









class Car:
    def __init__(self,index,type_of_car,type_of_move):#constructor that build the car
        self.index=index #the location of the car
        self.type_of_car=type_of_car # 1 if the car Occupying three slots and 0 if the car Occupying two slots
        self.type_of_move=type_of_move # 0 if the car move horizontal and 1 if the car move vertical

class Puzzle:
    size_table = 6
    number_of_nodes = 0
    visited = {}
    sum_herustic=0
    waiting_nodes = PriorityQueue()

    def search(self,herstic_id):
        start = time.time()

        while not self.waiting_nodes.empty():
            cur_node = self.waiting_nodes.get()[1]
            str_node = list_to_string(cur_node.array_puzle)
            while str_node in self.visited and not self.waiting_nodes.empty():
                cur_node = self.waiting_nodes.get()[1]
                str_node = list_to_string(cur_node.array_puzle)

            self.visited[str_node] = 1
            self.number_of_nodes+=1
            self.sum_herustic+=cur_node.herstic_val



            if cur_node.goal_state():

                timeExe = time.time() - start

                print("Time of execution:", timeExe)
                if timeExe<=time_limit:
                    temp_node=cur_node
                    last_move=[]
                    last_move.append('X')
                    last_move.append('R')
                    a=2

                    last_move.append(str(a))
                    list_moves=[]
                    count=1


                    while temp_node.parent!= None:

                        str_move=temp_node.move
                        if str_move[0]==last_move[0] and str_move[1]==last_move[1]:
                           count+=1
                        else:
                            str1=last_move[0]+last_move[1]+str(count)
                            list_moves.append(str1)
                            count=1
                            last_move=str_move
                        temp_node = temp_node.parent



                    if last_move!=None:
                        str1 = last_move[0] + last_move[1] + str(count)
                        list_moves.append(str1)
                    temp_node=cur_node
                    max_depth=None
                    min_depth=None
                    size=0
                    sum_depth=0
                    while not self.waiting_nodes.empty():
                        temp_node = self.waiting_nodes.get()[1]
                        size+=1
                        sum_depth+=temp_node.depth
                        if max_depth==None or max_depth<temp_node.depth:
                            max_depth=temp_node.depth
                        if min_depth==None or min_depth>temp_node.depth:
                            min_depth=temp_node.depth

                    avg_depth=0
                    if size==0:
                        avg_depth=0
                        max_depth=cur_node.depth
                        min_depth=cur_node.depth
                    else:
                        avg_depth=sum_depth/size










                    count=len(list_moves)
                    temp_list=[]
                    for i in range(count-1,-1,-1):
                        print(list_moves[i], end=" ")
                        temp_list.append(list_moves[i])


                    with open("solution.txt", "a") as my_file:
                        my_file.write(" ".join(temp_list)+' ' )
                        my_file.write("\n")





                    penetrance=cur_node.depth/self.number_of_nodes
                    avg_h_value=self.sum_herustic/self.number_of_nodes
                    ebf=self.number_of_nodes**(1/cur_node.depth)
                    print('\n number of visited nodes:',self.number_of_nodes)
                    print('herstic_id: ', herstic_id)
                    print(' Penetrance: ',penetrance)
                    print('avg H value: ',avg_h_value)
                    print('EBF: ',ebf)
                    print('Max depth: ',max_depth)
                    print('Min depth: ', min_depth)
                    print('Average depth: ',avg_depth)
                    return self.number_of_nodes,herstic_id,penetrance,avg_h_value,ebf,max_depth,min_depth,avg_depth,timeExe




                else:
                    with open("solution.txt", "a") as my_file:
                        my_file.write("failed\n")
                    print('failed')

                return self.number_of_nodes,herstic_id,0,self.sum_herustic/self.number_of_nodes,self.number_of_nodes**(1/cur_node.depth),cur_node.depth,cur_node.depth,cur_node.depth,10
            else:


                    cur_node.calc_successors(herstic_id)


    @staticmethod
    def add_to_waitlist(current_node,herstic_id):
        Puzzle.waiting_nodes.put((current_node.score(herstic_id), current_node))

    def init_the_game(self, node_text,herstic_id): #create the init board the root
        cur_cars = {}
        array_board = list()
        for i in range(0, self.size_table):
            array_board.append(list(node_text[i * self.size_table:i * self.size_table + self.size_table]))
        for i in range(0, self.size_table):
            for j in range(0, self.size_table):
                if array_board[i][j] == '.':
                    continue
                if ('A' <= array_board[i][j] <= 'K' or array_board[i][j] == 'X') and array_board[i][j] not in cur_cars:
                    type_car = 0
                    if j == self.size_table - 1:
                        type_move = 1
                    elif j < self.size_table - 1:
                        if array_board[i][j] == array_board[i][j + 1]:
                            type_move = 0
                        else:
                            type_move = 1
                elif 'O' <= array_board[i][j] <= 'R' and array_board[i][j] not in cur_cars:
                    type_car = 1
                    if j >= self.size_table - 2:
                        type_move = 1
                    elif j < self.size_table - 2:
                        if array_board[i][j + 1] == array_board[i][j] and array_board[i][j] == array_board[i][j + 2]:
                            type_move = 0
                        else:
                            type_move = 1

                if array_board[i][j] not in cur_cars:
                    index = (i, j)
                    car = Car(index, type_car, type_move)
                    cur_cars[array_board[i][j]] = car
                   # self.number_of_nodes += 1
        current_node = Node(cur_cars, None, 0, array_board, None, None)
        print_array(array_board)
        if herstic_id==4:
            val_herstic = current_node.calc_herustic_valtemp()

        else:
            val_herstic = current_node.calc_herustic_val(herstic_id)
        current_node.herstic_val = val_herstic
        if herstic_id=='iterative_deepning':
            return Iterative_deepning.iterative_deepning(current_node)
        else:
            self.add_to_waitlist(current_node,herstic_id)
            return self.search(herstic_id)





time_limit=4
if __name__ == "__main__":
 with open("rh.txt" ,"r") as f:
        fileData = f.read()


        data = fileData.split("--- RH-input ---")
        data = data[1]
        data = data.split("--- end RH-input ---")
        data = data[0]


        games = data.split("\n")
        games = games[1:]
        sum_N=0
        sum_penetrance=0
        sum_avg_h_value=0
        sum_ebf=0
        sum_max_depth=0
        sum_min_depth=0
        sum_avg_depth=0
        sum_time_exe=0
        levels_nodesnumber=[0,0,0,0]
        levels_time_exe=[0,0,0,0]
        levels_penetrance=[0,0,0,0]
        levels_avg_h=[0,0,0,0]
        levels_ebf=[0,0,0,0]
        levels_mindepth=[0,0,0,0]
        levels_avg_avgdepth=[0,0,0,0]
        levels_maxdepth=[0,0,0,0]
        size=len(games)-1

        for i in range(0, size):
            print("-------------------------------------------------------------------------------------------------------")
            print("Problem:" ,str(i+1))
            currentData =games[i]
            puzzle = Puzzle()
            if i+1==14:
                N, herstic_id, penetrance, avg_h_value, ebf, max_depth, min_depth, avg_depth, time_exe = puzzle.init_the_game(currentData, 3)

            else:
                N,herstic_id,penetrance,avg_h_value,ebf,max_depth,min_depth,avg_depth,time_exe=puzzle.init_the_game(currentData,9)
            j=int(i/10)
            levels_nodesnumber[j]+=N
            levels_time_exe[j]+=time_exe
            levels_penetrance[j]+=penetrance
            levels_ebf[j]+=ebf
            levels_avg_h[j]+=avg_h_value
            levels_mindepth[j]+=min_depth
            levels_avg_avgdepth[j]+=avg_depth
            levels_maxdepth[j]+=max_depth
            sum_N+=N
            sum_penetrance+=penetrance
            sum_avg_h_value+=avg_h_value
            sum_ebf+=ebf
            sum_max_depth+=max_depth
            sum_min_depth+=min_depth
            sum_avg_depth+=avg_depth
            sum_time_exe+=time_exe

        print("-------------------------------------------------------------------------------------------------------")
        print(herstic_id)
        print('average number of vistid nodes: ',sum_N/size)
        print('average time of execution: ',sum_time_exe/size)
        print('average penetrance: ',sum_penetrance/size)
        print('average ebf: ',sum_ebf/size)
        print('average H value: ', sum_avg_h_value/size)
        print('average max depth: ',sum_max_depth/size)
        print('average min depth: ',sum_min_depth/size)
        print('average of average depth: ',sum_avg_depth/size)
        print("-------------------------------------------------------------------------------------------------------")
        levels=['beginer','intermdate','advanced','expert']
        for i in range(0,len(levels)):
            print('average number of vistid nodes for ',levels[i], 'is: ', levels_nodesnumber[i]/10)
            print('average time of execution for ', levels[i], 'is: ', levels_time_exe[i] / 10)
            print('average penetrance for ', levels[i], 'is: ', levels_penetrance[i] / 10)
            print('average ebf for ', levels[i], 'is: ', levels_ebf[i] / 10)
            print('average H value for ', levels[i], 'is: ', levels_avg_h[i] / 10)
            print('average min depth for ', levels[i], 'is: ', levels_mindepth[i] / 10)
            print('average of average depth for ', levels[i], 'is: ', levels_avg_avgdepth[i] / 10)
            print('average max depth for ', levels[i], 'is: ', levels_maxdepth[i] / 10)







