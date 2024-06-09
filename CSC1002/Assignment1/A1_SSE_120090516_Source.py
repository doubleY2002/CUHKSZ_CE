a=[
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]
]
n=0
moves=""
move=[]
x=n-1
y=n-1
def print_board():
    global a
    for i in range(n):
        for j in range(n):
            if(j!=n-1):
                print(a[i][j],end="\t")
            else :
                print(a[i][j])

def change(step):#0left 1right 2up 3down
    global a,x,y
    if step==0:
        a[x][y],a[x][y-1]=a[x][y-1],a[x][y]
        y-=1
    elif step==1:
        a[x][y],a[x][y+1]=a[x][y+1],a[x][y]
        y+=1
    elif step==2:
        a[x][y],a[x-1][y]=a[x-1][y],a[x][y]
        x-=1
    elif step==3:
        a[x][y],a[x+1][y]=a[x+1][y],a[x][y]
        x+=1

def check_next(la=-1):
    S=[]
    global x,y
    if y>0 and la!=1:
        S.append(0)
    if y<n-1 and la!=0:
        S.append(1)
    if x>0 and la!=3:
        S.append(2)
    if x<n-1 and la!=2:
        S.append(3)
    return S

def str_to_int(x):
    for i in range(4):
        if x==move[i]:
            return i
    return -1
    
def create_board():
    global a,x,y
    for i in range(n):
        for j in range(n):
            a[i][j]=str(i*n+j+1)
    a[n-1][n-1]=" "
    check_end()
    steps=random.randint(n**2,110)
    x=n-1
    y=n-1
    la=-1
    while steps:
        steps-=1
        S=check_next(la)
        la=random.randint(0,3)
        if la in S:
            change(la)
        else:
            steps+=1

def init():
    global n,moves,move
    n=eval(input("Enter the desired dimension of the puzzle > "))
    moves=input("Enter the four letters used for left, right, up and down directions > ")
    move=moves.split()
    create_board()
    print_board()

def check_end():
    global a
    for i in range(n):
        for j in range(n):
            if i!=n-1 or j!=n-1:
                if a[i][j]==' ':
                    return True
                elif int(a[i][j])!=i*n+j+1:
                    return True
    return False

def get(S):
    global move
    orz=""
    direction=["left","right","up","down"]
    for i in range(len(S)):
        if(i!=len(S)-1):
            orz=orz+direction[S[i]]+"-"+move[S[i]]+","
        else:
            orz=orz+direction[S[i]]+"-"+move[S[i]]
    return orz

def solve():
    cnt=0
    global x,y
    while check_end():
        S=check_next()
        s=str_to_int(input("Enter your move (%s): "%get(S)))
        while s not in S:
            s=str_to_int(input("Enter a right move (%s): "%get(S)))
        change(s)
        print_board()
        cnt+=1
        la=s
    print("Congratulations! You solved the puzzle in %d moves!"%cnt)

def start_new():
    norq=input("Enter ‘n’ to start a new game or enter ‘q’ to end the game >")
    if norq=="q":
        return True
    else:
        return False

print("Here is an interactive sliding puzzle game. You will be given a board. The board has an empty space where an adjacent tile can be slid to. The objective of the game is to rearrange the tiles into a sequential order by their numbers (left to right, top to bottom) by repeatedly making sliding moves (left, right, up or down). When  all numbers appear sequentially, ordered from left to right, top to bottom, and the empty space is on the rightest and bottom, the game ends.")
import random
while True:
    init()
    solve()
    if start_new():
        break
print("Good-bye!")