class Node():

    def __init__(self, cell):
        self.cell = cell
        self.next = None

########################################################



class Linked_List():

    def __init__(self):
        self.head = None


    def append_node(self, node):
        new_node = Node(node)

        if self.head is None:
            self.head = new_node
            return
        else:
            current = self.head
            while True:
                if current.next is None:
                    current.next = new_node
                    break
                current = current.next


    def print_list(self):
        list = []
        current = self.head
        while current is not None:
            list.append(current.cell.get_loc())
            current = current.next
        print(list)



    def get_path(self):
        list = []
        current = self.head
        while current is not None:
                list.append((current.cell.get_loc(),current.cell.get_parent()))
                current = current.next
        return list


    def has_location(self, child):
        current = self.head
        while current.next is not None:
            current = current.next
            if current.cell.get_loc() == child:
                return True

        return False

    def get_location(self, i):
        count = 0
        current = self.head
        while current is not None:
            current = current.next
            if count == i:
                return current.cell.get_loc()
            else:
                count += 1
                current = current.next



    def get_last_g(self):
        current = self.head
        while current is not None:
            if current.next is None:
                return current.cell.get_g()
            else:
                current = current.next




    def length(self):
        current = self.head
        count = 1
        while current.next is not None:
            current = current.next
            count += 1
        return count



