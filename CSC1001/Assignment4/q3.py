class Hanoi:
    def __init__(self,num=1,fr='A',he='B',to='C'):
        self.num=num
        self.fr=fr
        self.he=he
        self.to=to
    
    def print(self):
        print(self.fr,'-->',self.to)

class Stack:
    def __init__(self):
        self.__data=list()
    
    def is_empty(self):
        return len(self.__data)==0
    
    def push(self,e=Hanoi()):
        self.__data.append(e)

    def pop(self):
        return self.__data.pop()

def Hanoitower(n):
    S=Stack()
    S.push(Hanoi(n))
    while S.is_empty()==False:
        now=S.pop()
        if now.num==1: now.print()
        else:
            num=now.num-1
            S.push(Hanoi(num,now.he,now.to,now.fr))
            S.push(Hanoi(  1,now.fr,now.to,now.he))
            S.push(Hanoi(num,now.fr,now.he,now.to))

def main():
    n=eval(input("Please enter the number of disks:"))
    Hanoitower(n)

main()