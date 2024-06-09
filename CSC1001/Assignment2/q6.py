f=0
board=list()
def kill(pos,flag):
    global board
    for i in range((pos//8)*8,(pos//8+1)*8):##row
        board[i]+=flag
    for i in range(0,8):##column
        board[i*8+pos%8]+=flag
    i=pos
    while i-7>0 and i%8!=7:##right up
        i-=7
        board[i]+=flag
    i=pos
    while i+7<63 and i%8!=0:##left down
        i+=7
        board[i]+=flag
    i=pos
    while i+9<64 and i%8!=7:##right down
        i+=9
        board[i]+=flag
    i=pos
    while i-9>0 and i%8!=0:##left up
        i-=9
        board[i]+=flag
def getqueen(pos,cnt):
    global board,f
    kill(pos,1)
    board[pos]=-1
    if cnt==8:
        f=1
        return
    for i in range(pos,64):
        if board[i]==0:
            getqueen(i,cnt+1)
        if f==1: 
            return
    kill(pos,-1)
    board[pos]=0
def print_board():
    global board
    for i in range(64):
        if board[i]==-1: print("|Q",end="")
        else: print("| ",end="")
        if i%8==7:print("|")
for i in range(64):
    board.append(0)
getqueen(1,1)
print_board()