import copy

class Priority_Queue():

    def __init__(self):
        self.queue = [0]
        self.length = 0
        self.small_g  = False       # set to True if we want small-g as tie breaker



    def get_length(self):
        return self.length


    def is_empty(self):
        if len(self.queue) == 0:
            return True
        else:
            return False


    # Add new cell to end of queue, increment length, call method to fix heap order
    def add_child(self, cell):
        self.queue.append(cell)
        self.length += 1
        self.fix_heap_up()


    def fix_heap_up(self):
        i = self.length
        # Only need to check if queue order is maintained if more than one item in the list
        while i > 1:
            # If the new child is smaller than parent, they need to be swapped
            if self.queue[i].get_f() < self.queue[i // 2].get_f():
                temp_node = self.queue[i // 2]
                self.queue[i // 2] = self.queue[i]
                self.queue[i] = temp_node
            ## Add implementation for deciding on a node if the f values are the same
            if  self.queue[i].get_f() ==  self.queue[i // 2].get_f():
                if self.small_g:
                    # If the g value of the node adding is less than the root, they need to be swapped
                    # If not, then nothing needs to happen
                    if self.queue[i].get_g() < self.queue[i // 2].get_g():
                        temp_node = self.queue[i // 2]
                        self.queue[i // 2] = self.queue[i]
                        self.queue[i] = temp_node
                #Only do this if small_g is set to false
                else:
                    if self.queue[i].get_g() > self.queue[i // 2].get_g():
                        temp_node = self.queue[i // 2]
                        self.queue[i // 2] = self.queue[i]
                        self.queue[i] = temp_node
            ## Complete implementation of big/small g

            i = i//2


    def fix_heap_down(self):
        i = 1
        while (i * 2) <= self.length:
            smallest_child = self.get_smallest_child(i)
            if self.queue[smallest_child].get_f() < self.queue[i].get_f():
                temp_node = self.queue[smallest_child]
                self.queue[smallest_child] = self.queue[i]
                self.queue[i] = temp_node
            ## BEGIN IMPLEMENTATION OF SMALL G, BIG G
            elif self.queue[smallest_child].get_f() == self.queue[i].get_f():
                if self.small_g:
                    # If the g value of the node adding is less than the root, they need to be swapped
                    # If not, then nothing needs to happen
                    if self.queue[smallest_child].get_g() < self.queue[i].get_g():
                        temp_node = self.queue[smallest_child]
                        self.queue[smallest_child] = self.queue[i]
                        self.queue[i] = temp_node
                    # Only do this if small_g is set to false
                else:
                    if self.queue[smallest_child].get_g() > self.queue[i].get_g():
                        temp_node = self.queue[smallest_child]
                        self.queue[smallest_child] = self.queue[i]
                        self.queue[i] = temp_node
            ## Complete implementation of big/small g

            i = smallest_child


    # Get the root of the queue, move last node to the root, reduce length by 1, call method to fix heap
    def get_root(self):

        if self.length == 1:
            root = self.queue[1]
            self.length -= 1
            self.queue.pop()
            return root

        root = copy.deepcopy(self.queue[1])

        self.queue[1].set_g(self.queue[self.length].get_g())
        self.queue[1].set_h(self.queue[self.length].get_h())
        self.queue[1].set_f(self.queue[self.length].get_f())
        self.queue[1].set_loc(self.queue[self.length].get_loc())
        self.queue[1].set_parent(self.queue[self.length].get_parent())


        self.length -= 1
        self.queue.pop()
        self.fix_heap_down()

        return root


    def get_smallest_child(self, i):
        # Return that node if it is the last one
        if ((i * 2) + 1) > self.length:
            return (i * 2)
        ## ADDING BIG/SMALL g implementation
        if self.queue[i * 2].get_f() == self.queue[(i * 2) + 1].get_f():
            if self.small_g:
                if self.queue[i*2].get_g() < self.queue[(i * 2) +1 ].get_g():
                    return (i * 2)
                else:
                    return (i * 2) + 1
            # Only do this if small_g is set to false
            else:
                if self.queue[i*2].get_g() < self.queue[(i * 2) +1 ].get_g():
                    return (i * 2) + 1
                else:
                    return (i * 2)
        ## Complete implementation of big/small g

        if self.queue[i*2].get_f() < self.queue[(i*2) + 1].get_f():
            return (i * 2)
        else:
            return (i * 2) + 1




