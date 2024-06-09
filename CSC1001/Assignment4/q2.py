class Node:
    def __init__(self, element=0, pointer=None):
        self.element=element
        self.pointer=pointer

class SinglyLinkedList:
    def __init__(self,head=Node(),end=None):
        self.head=head
        self.end=end
    
    def insert(self, data=0):
        if self.head.pointer==None:
            self.tail=Node(data)
            self.head.pointer=self.tail
        else:
            self.tail.pointer=Node(data)
            self.tail=self.tail.pointer
    
    def quick_sort(self,node):
        if node.pointer==self.end: return
        key=node.pointer
        now=key.pointer
        end=self.end
        while now!=end:
            if now.element<key.element:
                element=now.element
                if key.pointer==now:
                    now.element=key.element
                    key.element=element
                else:
                    now.element=key.pointer.element
                    key.pointer.element=key.element
                    key.element=element
                key=key.pointer
            now=now.pointer
        self.end=key
        self.quick_sort(node)
        self.end=end
        self.quick_sort(key)
    
    def print(self):
        node=self.head.pointer
        while node!=None:
            print(node.element)
            node=node.pointer


def test():
    L=SinglyLinkedList()
    nums=[49, 38, 65, 97, 23, 22, 76, 1, 5, 8, 2, 0, -1, 22]
    for i in range(len(nums)):
        L.insert(nums[i])
    #L.print()
    L.quick_sort(L.head)
    L.print()

test()