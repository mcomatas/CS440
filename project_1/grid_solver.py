# Created: 7/09/2020
# Authors: Douglas Gromek, Michael Comatas, Xiaoyu Sun
# Purpose: Create, Store, and solve grid mazes



import numpy as np
import random
import time
import matplotlib.pyplot as plt

from priority_queue import Priority_Queue
from cell import Cell
from visited_node import Linked_List

run_number = 1
grids_solved: int = 0
grids_failed: int = 0

# Variable numbers
row = 101  # Will be set to 101 for actual testing
col = 101
per = [0.7, 0.3]  # approx. percentage of blocked cells is second number
num_arrays = 1   # how many arrays to generate
run_count = 0


create_new = True  #Decide if you want to make new arrays or not
adaptive_h = True
backward = False  # set True to perform backward A*

start_time = time.time()


if create_new:
    main_file  = open("Data/run_" + str(run_number) + ".txt", "w")
    main_file.write("\n\n")
else:
    main_file = open("Data/run_" + str(run_number) + ".txt", "a")
    main_file.write("\n\n")


start_time = time.time()





# Make each array and save it to a .txt file
def create_array(num):
    count = 1
    for i in range(num):
        arr = np.random.choice([0, 1], size=(row, col), p = per ).astype('uint8')

        np.savetxt('Arrays/array_' + str(count) + ".txt", arr, fmt = '%d', delimiter=',')

        plt.figure()
        plt.imshow(arr, cmap=plt.cm.binary, interpolation='nearest')
        plt.xticks([]), plt.yticks([])
        plt.savefig("Grids/grid_" + str(count))

        #plt.show()
        count += 1
        plt.close()





# Load in arrays from txt files
def load_array(count):
    array = np.loadtxt("Arrays/array_" +  str(count+1) + ".txt", delimiter=',').astype(int)
    return array


# Randomly generates the start and the goal positions of the array, then returns them
def get_start_stop(arr):
    start_pos = (random.randint(0, row - 1), random.randint(0, col - 1))
    goal_pos = (random.randint(0, row - 1), random.randint(0, col - 1))

    while arr[start_pos] == 1 or arr[goal_pos] == 1:
        start_pos = (random.randint(0, row - 1), random.randint(0, col - 1))
        goal_pos = (random.randint(0, row - 1), random.randint(0, col - 1))

    return [start_pos, goal_pos]



# Get children of current position but not start
def get_children(array, current_node, child_list, expanded_list, blocked_list):

    curr = current_node.get_loc()

    #Generate all 4 children, will filter later
    temp_list =  []
    temp_list.append((curr[0] + 1, curr[1]))
    temp_list.append((curr[0] - 1, curr[1]))
    temp_list.append((curr[0], curr[1] + 1))
    temp_list.append((curr[0], curr[1] - 1))

    removed = 0

    # Loop to remove out of bounds or blocked children
    for i in range(len(temp_list)):

        child = temp_list[i - removed]

        # removed "check_blocked(blocked_list, array,child) or "
        # Get rid of any potential oob or blocked children or visisted children
        if check_oob(child) or check_expanded(expanded_list, child) or check_current_blocked_list(blocked_list, child):
            temp_list.remove(child)
            removed += 1

    # return all remaining children
    return temp_list


#check if a location in current blocked list
def check_current_blocked_list (blocked_list, location):
    for i in blocked_list:
        if location == i:
            return True
    return False


# Get children of current position
def get_start_states_children(array, current_node, child_list, expanded_list, blocked_list):

    curr = current_node.get_loc()

    #Generate all 4 children, will filter later
    temp_list =  []
    temp_list.append((curr[0] + 1, curr[1]))
    temp_list.append((curr[0] - 1, curr[1]))
    temp_list.append((curr[0], curr[1] + 1))
    temp_list.append((curr[0], curr[1] - 1))

    removed = 0

    # Loop to remove out of bounds or blocked children
    for i in range(len(temp_list)):

        child = temp_list[i - removed]

        # Get rid of any potential oob or blocked children or visisted children
        if check_oob(child) or check_blocked(blocked_list, array,child) or check_expanded(expanded_list, child):
            temp_list.remove(child)
            removed += 1

    # return all remaining children
    return temp_list



# Method to child if child is OOB
def check_oob(child):
    # Get rid of any children out of array bounds
    if child[0] < 0 or child[1] < 0 or child[0] >= row or child[1] >= col:
        return True
    else:
        return False


# Method to check if a child is blocked, cell value = 1
def check_blocked(blocked_list, array, child):
    #Get rid of any blocked children and add to blocked list

    #If the value of the array at the child cell is a "1", add it to the blocked list
    if array[child[0]][child[1]] == 1:
        blocked_list.append(child)
        return True
    else:
        return False



# Check if the child has already been visited
def check_expanded(expanded_list, child):

    # Get rid of any children that have already been visited
    if expanded_list.has_location(child):
        return True
    else:
        return False




# Calculate all the info needed to create a cell object for each child that we will be
# adding to the child list
def add_children(child, child_list, current_node, goal_node, adaptive_heuristic, final_path ):

    h = abs(goal_node.get_loc()[0] - child[0] ) + abs(goal_node.get_loc()[1] - child[1] )
    g = current_node.get_g() + 1
    f = g  + h
    loc  = child
    parent = current_node.get_loc()

    # Only do if we want adaptive heuristics
    if adaptive_h:
        # Only do if the child being added is in the final path
        if child in final_path:
            h = adaptive_heuristic - g


    #Create the cell
    new_child = Cell(f,g,h,loc,parent)


    # Add the cell to the child list
    child_list.add_child(new_child)


# expanded_list: in this single A*, the nodes we have expanded
# child_list: in this single A*, the nodes we have visited
# we only use this to give out the presumed best solution, the agent doesn't move at all
def a_star_solver(array, start_node, goal_node, child_list, expanded_list, blocked_list, adaptive_heuristic, final_path):

    new_run = 0

    while child_list.get_length() != 0:
        # Update the current node
        current_node = child_list.get_root()

        if new_run != 0:
            while expanded_list.has_location(current_node.get_loc()):
                if child_list.get_length() == 0:
                    return []
                else:
                    current_node = child_list.get_root()

        # Add current node to the expanded list
        expanded_list.append_node(current_node)

        new_run += 1

        #print("EXPANDED NODE: " + str(current_node.get_loc())   + "      ITS PARENT: " +  str(current_node.get_parent()))
        #print("expanded list:")
        #print("LENGHTH OF EXPANDED IS:" + str(expanded_list.length()))
        #expanded_list.print_list()

        # check if current node is goal node
        # if we reached the goal, meaning that based on our current knowledge
        # we have found the path, though the path could contains obstacle by now
        if current_node.get_loc() == goal_node.get_loc():
            return success_message(child_list, expanded_list)        
        else:
            # get_start_states_children: filter out blocked cell that we can see from current state
            # get_children: do not filter out blocked cell in the blocked list
            if current_node.get_loc() == start_node.get_loc():
                new_children = get_start_states_children(array, current_node, child_list, expanded_list,blocked_list)
            else:
                new_children = get_children(array, current_node, child_list, expanded_list,blocked_list)
   

        #For each new child, create its "cell" object and calculate all costs/values for it
            for i in new_children:
                add_children(i,child_list, current_node, goal_node,adaptive_heuristic, final_path)

    if child_list.get_length() == 0:
        return [] 

'''    
        # Update the current node
        current_node = child_list.get_root()

        # Go to success method if we have reached our destination
        if current_node.get_loc() == goal_node.get_loc():
            success_message(child_list, expanded_list)
            return   

        # Add current node to the expanded list
        expanded_list.append_node(current_node)
        print("EXPANDED NODE: " + str(current_node.get_loc())   + "      ITS PARENT: " +  str(current_node.get_parent()))

        # get new children and continue
        new_children = get_children(array, current_node, child_list, expanded_list,blocked_list)

        #For each new child, create its "cell" object and calculate all costs/values for it
        for i in new_children:
            add_children(i,child_list, current_node, goal_node)



        if child_list.get_length() == 0:
            failure_message()
            return

'''



def failure_message():
    global grids_failed
    grids_failed += 1
    main_file.write("\nArray " + str(run_count + 1) + ":     NOT SOLVABLE")

    print("FAILURE")
    print("TOTAL RUNTIME WAS: --- %s seconds ---" % (time.time() - start_time))


def success_message(child_list, expanded_list):

    path = []

    # each element in my_tuple: (node (r,c), parent(r,c))
    my_tuple = expanded_list.get_path()

    my_tuple.reverse()

    count = 0
    #Loop through the entire list
    for i in my_tuple:

        #Append location and then escape if we are the last element
        # i[0] is that nodes location, while i[1 ] is that nodes parent location
        if i[1] == None:                    # start node
            path.append(i[0])
            continue
        elif count == 0:                    # we are at the goal node
            parent = i[1]
            path.append(i[0])
            count += 1
        else:                               # appends to the path only of the parent is equal to that node
            if i[0] == parent:
                path.append(i[0])
                parent = i[1]


    path.reverse()
    #print("presumued shortest path:")
    #print(path)
    return path



''' 
    print("THE TOTAL NUMBER OF MOVES REQUIRED WAS: " + str(len(path) -  1))

    print("TOTAL RUNTIME WAS: --- %s seconds ---" % (time.time() - start_time))
'''


# Make new arrays if it is needed
if create_new == True:
    create_array(num_arrays)


# Create, save, and then load each array. Pass each array into the solver
while run_count < num_arrays:

    if create_new:

        #Load each array
        array = load_array(run_count)

        #get start and goal positions
        pos = get_start_stop(array)

        #################################### ADDING SAVE DATA STUFF

        # Create a txt file that will store all information about the array
        data_file = open("Data/array_" + str(run_count + 1) + ".txt", "w")

        # Write the start and goal positions into the file
        data_file.write(str(pos[0]) + ":  START NODE")
        data_file.write(" \n")
        data_file.write(str(pos[1]) + ":   GOAL NODE")

        data_file.write(" \n")
        data_file.write(" \n")

        ################################################################
        #initialize expanded_list
        expanded_list = Linked_List()

        #Create a priority queue for the child list
        child_list = Priority_Queue()

        # Create the start and goal nodes
        start_node = Cell(0,0,0,pos[0],None)
        goal_node =  Cell(0,0,0,pos[1],None)


        # initialize the child_list, add the start node to it
        if backward == False:
            child_list.add_child(start_node)
        else:
            child_list.add_child(goal_node)


        # Create the initial blocked list
        blocked_list = []


        print("STARTING SOLVER")
        print(array)
        print("START: " + str(start_node.get_loc()) + "   GOAL:  " + str(goal_node.get_loc()))

        # the final path
        final_path = []

        # do we need to repeat A* ?
        need_repeat_A_star = False

        #Adaptive HEURISTIC STUFF
        new_run = 0
        adaptive_heuristic = 0

        # Begin the A* Search
        # This is the real implement, where the agent will move along the path
        while child_list.get_length() != 0 or need_repeat_A_star == True:
            need_repeat_A_star = False
            # temp path: store the path that the A* gives out
            temp_path = []

            # temp_path: the presumed best solution, a list of tuple
            # backward: the decide making boolean
            # ==========Here we decided if we are using backward or forward ===========    
            if backward == False:
                temp_path = a_star_solver(array, start_node, goal_node,child_list,expanded_list,blocked_list, adaptive_heuristic, final_path)
            else:
                #here means that we do use backward
                #we need to swap: start should be the goal, goal should be start
                #then we need to reverse the temp_path, since we move the agent from initally start 
                temp_path = a_star_solver(array, goal_node, start_node, child_list,expanded_list,blocked_list, adaptive_heuristic, final_path)
                print("Backward path: from goal to start")
                print(temp_path)
                temp_path.reverse()

            # =========================================================================    


            new_run += 1
            ####################### ADAPTIVE HEURISTIC ############################################

            if new_run != 0:
                #ADAPTIVE HEURISTICS. SAVE THE G VALUE OF THE GOAL NODE AFTER THE FIRST RUN
                if len(temp_path) != 0 :
                    # print(">>>>>>>>>>>>>THE G VALUE OF THE GOAL IS : " + str(expanded_list.get_last_g()))
                    adaptive_heuristic = len(temp_path)-1


            ########################################################################################


            # is there a path the agent can move along?
            if len(temp_path) == 0:     # impossible to solve
                failure_message()
                break
            else:                       # we have a new presumed best solution
                print('new presumed best solution:')
                print(temp_path)


            # Traverse the presumed best solution, i is a location, stored as tuple
            for i in temp_path:

                # check any of the cells in the path is blocked
                if check_blocked(blocked_list, array, i):         # oops we encountered an obstacle
                    need_repeat_A_star = True                     # set the detector to True

                    # Now we need to find the parent of the obsticle
                    # restart A* from the node, namely the new start_node

                    print("here is where we have the OBSTACLE")
                    print(i)
                    print("The final_path by far we got")
                    print(final_path)

                    # implement：
                    # find the previous location of the obstacle in the temp_path
                    # use that location to create a new start, (f,g,h,loc,parent)
                    # start_node = expanded_list.get_parent(i)

                    parent_of_obstacle = ()

                    # Here we traverse the presumed path to find the parent of the obsticle
                    # set it to be new start state
                    for w in range(len(temp_path)):
                        if i == temp_path[w]:
                            parent_of_obstacle = temp_path[w-1]

                    # create the new start cell
                    temp_g = 0
                    temp_h = abs(goal_node.get_loc()[0] - parent_of_obstacle[0] ) + abs(goal_node.get_loc()[1] - parent_of_obstacle[1] )
                    temp_f = temp_g + temp_h
                    start_node = Cell(temp_f,0,temp_h,parent_of_obstacle,None)

                    print("new start node:")
                    print(start_node.get_loc())
                    break

                else:                           # i is un-blocked, we can safely add it to final path
                    final_path.append(i)

                    # we alos need to check the children among the path see if they are blocked
                    # but only the cells before the blocked cell in the path
                    temp_check_child_list = []
                    temp_check_child_list.append((i[0]+1, i[1]))
                    temp_check_child_list.append((i[0]-1, i[1]))
                    temp_check_child_list.append((i[0], i[1] + 1))
                    temp_check_child_list.append((i[0], i[1] - 1))

                    # remember the blocked cells we have visited
                    for t in temp_check_child_list:
                        if not check_oob(t):
                            #print("here is the t")
                            #print(t)
                            check_blocked(blocked_list, array, t)

                    #print("here the newest BLOCKED CELLS:")
                    #print(blocked_list)



            # Do we need to do A star again?

            if need_repeat_A_star == True:          #sadly we got wrong path, lets find a new one
                expanded_list = Linked_List()       # have a new expanded list
                print("========== Restart the A star search ==========")
                #expanded_list.print_list()
                child_list = Priority_Queue()       # have a new child list, we need to update how the data

                # here we check if we are using backward
                if backward == False:
                    child_list.add_child(start_node)    # initialize the child list with new start cell
                else:
                    child_list.add_child(goal_node)
            else:                                   #output the solution
                print("THE CUMULATIVE FINAL PATH")
                print(final_path)
                print("THE TOTAL NUMBER OF MOVES REQUIRED WAS: " + str(len(final_path) -  1))
                print("TOTAL RUNTIME WAS: --- %s seconds ---" % (time.time() - start_time))
                grids_solved += 1
                print(str(grids_solved))

                main_file.write("\nArray " + str(run_count + 1) + ": Solved in: " + str(len(final_path) - 1) + " moves")
                break

        run_count += 1
    else:
        # Load array
        array = load_array(run_count)

        ############################################### SAVE DATA STUFF
        # Open folder containing the start and goal nodes
        data_file = open("Data/array_" + str(run_count + 1) + ".txt")

        # Load the starting and the goal positions lines
        start = (data_file.readline())
        goal = (data_file.readline())

        # Split the coordinates away from the text
        start_tuple = start.split(":")
        goal_tuple = goal.split(":")

        # Convert the coordinates back into integer tuples
        int_start = eval(start_tuple[0])
        int_goal = eval(goal_tuple[0])

        #############################################################

        # initialize expanded_list
        expanded_list = Linked_List()

        # Create a priority queue for the child list
        child_list = Priority_Queue()

        # Create the start and goal nodes
        start_node = Cell(0, 0, 0, int_start, None)
        goal_node = Cell(0, 0, 0, int_goal, None)


        # initialize the child_list, add the start node to it
        if backward == False:
            child_list.add_child(start_node)
        else:
            child_list.add_child(goal_node)
  

        # Create the initial blocked list
        blocked_list = []

        print("STARTING SOLVER")
        print(array)
        print("START: " + str(start_node.get_loc()) + "   GOAL:  " + str(goal_node.get_loc()))

        # the final path
        final_path = []

        # do we need to repeat A* ?
        need_repeat_A_star = False


        #ADAPTIVE HEURISTIC STUFF
        new_run = 0
        adaptive_heuristic = 0

        # Begin the A* Search
        # This is the real implement, where the agent will move along the path
        while child_list.get_length() != 0 or need_repeat_A_star == True:
            need_repeat_A_star = False
            # temp path: store the path that the A* gives out
            temp_path = []


            # temp_path: the presumed best solution, a list of tuple
            # backward: the decide making boolean
            # ==========Here we decided if we are using backward or forward ===========    
            if backward == False:
                temp_path = a_star_solver(array, start_node, goal_node,child_list,expanded_list,blocked_list, adaptive_heuristic, final_path)
            else:
                #here means that we do use backward
                #we need to swap: start should be the goal, goal should be start
                #then we need to reverse the temp_path, since we move the agent from initally start 
                temp_path = a_star_solver(array, goal_node, start_node, child_list,expanded_list,blocked_list, adaptive_heuristic, final_path)
                print("Backward path: from goal to start")
                print(temp_path)
                temp_path.reverse()

            # =========================================================================    

            new_run += 1


            ####################### ADAPTIVE HEURISTIC ############################################

            if new_run != 0:
                # ADAPTIVE HEURISTICS. SAVE THE G VALUE OF THE GOAL NODE AFTER THE FIRST RUN
                if len(temp_path) != 0:
                    #print(">>>>>>>>>>>>>THE G VALUE OF THE GOAL IS : " + str(expanded_list.get_last_g()))
                    adaptive_heuristic = len(temp_path)-1



            ########################################################################################

            # is there a path the agent can move along?
            if len(temp_path) == 0:  # impossible to solve
                failure_message()
                break
            else:  # we have a new presumed best solution
                print('new presumed best solution:')
                print(temp_path)

            # Traverse the presumed best solution, i is a location, stored as tuple
            for i in temp_path:

                # check any of the cells in the path is blocked
                if check_blocked(blocked_list, array, i):  # oops we encountered an obstacle
                    need_repeat_A_star = True  # set the detector to True

                    # Now we need to find the parent of the obsticle
                    # restart A* from the node, namely the new start_node

                    print("here is where we have the OBSTACLE")
                    print(i)
                    print("The final_path by far we got")
                    print(final_path)
                    
                    # implement：
                    # find the previous location of the obstacle in the temp_path
                    # use that location to create a new start, (f,g,h,loc,parent)
                    # start_node = expanded_list.get_parent(i)

                    parent_of_obstacle = ()

                    # Here we traverse the presumed path to find the parent of the obsticle
                    # set it to be new start state
                    for w in range(len(temp_path)):
                        if i == temp_path[w]:
                            parent_of_obstacle = temp_path[w - 1]

                    # create the new start cell
                    temp_g = 0
                    temp_h = abs(goal_node.get_loc()[0] - parent_of_obstacle[0]) + abs(
                        goal_node.get_loc()[1] - parent_of_obstacle[1])
                    temp_f = temp_g + temp_h
                    start_node = Cell(temp_f, 0, temp_h, parent_of_obstacle, None)

                    print("new start node:")
                    print(start_node.get_loc())
                    break

                else:  # i is un-blocked, we can safely add it to final path
                    final_path.append(i)

                    # we alos need to check the children among the path see if they are blocked
                    # but only the cells before the blocked cell in the path
                    temp_check_child_list = []
                    temp_check_child_list.append((i[0] + 1, i[1]))
                    temp_check_child_list.append((i[0] - 1, i[1]))
                    temp_check_child_list.append((i[0], i[1] + 1))
                    temp_check_child_list.append((i[0], i[1] - 1))

                    # remember the blocked cells we have visited
                    for t in temp_check_child_list:
                        if not check_oob(t):
                            # print("here is the t")
                            # print(t)
                            check_blocked(blocked_list, array, t)

            # Do we need to do A star again?

            if need_repeat_A_star == True:  # sadly we got wrong path, lets find a new one
                expanded_list = Linked_List()  # have a new expanded list
                print("========== Restart the A star search ==========")
                #expanded_list.print_list()
                child_list = Priority_Queue()  # have a new child list, we need to update how the data

                # here we check if we are using backward
                if backward == False:
                    child_list.add_child(start_node)    # initialize the child list with new start cell
                else:
                    child_list.add_child(goal_node)     # initialize the child list with goal cell


            else:  # output the solution

                print("THE CUMULATIVE FINAL PATH")
                print(final_path)
                print("THE TOTAL NUMBER OF MOVES REQUIRED WAS: " + str(len(final_path) - 1))
                print("TOTAL RUNTIME WAS: --- %s seconds ---" % (time.time() - start_time))

                grids_solved += 1
                print(str(grids_solved))

                main_file.write("\nArray " + str(run_count + 1) + ": Solved in: " + str(len(final_path) - 1) + " moves")

                break
        run_count += 1




main_file.write("\n\nSolver Complete: \n\n")
main_file.write("\tTotal Number of grids: " + str(num_arrays))
main_file.write("\n\tTotal number of grids SOLVED: " +  str(grids_solved))
main_file.write("\n\tTotal number of grids FAILED: " +  str(grids_failed))
main_file.write("\nTOTAL RUNTIME WAS: --- %s seconds ---" % (time.time() - start_time))



main_file.close()






