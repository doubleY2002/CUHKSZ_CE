class Node:
    def __init__(self, element=0, pointer=None):
        self.element=element
        self.pointer=pointer

class SinglyLinkedList:
    def __init__(self,head=Node()):
        self.head=head
    
    def insert(self, data=0):
        self.head=Node(data,self.head)
    
    def recursive_count(self,node):
        if node.pointer==None:
            return 1
        else:
            return 1+self.recursive_count(node.pointer)

def test():
    L=SinglyLinkedList(Node(7))
    L.insert(5)
    L.insert(8)
    L.insert(3)
    print(L.recursive_count(L.head))

#test()