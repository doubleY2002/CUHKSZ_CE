import random

class ecosystem:
    def __init__(self,fish=1,bear=1,river=['N','F','B']):
        self.fish=fish
        self.bear=bear
        self.river=river
    def ck(self):
        a=self.river[0]
        if a=="N":
            return False
        for i in self.river:
            if i!=a:
                return False
        return True
    def solve(self,step=1):
        tot=0
        while tot<step:
            pos=list()
            none=list()
            move=[-1,0,1]
            left=[0,1]
            right=[-1,0]
            count=0
            for i in self.river:
                if i=="F" or i=="B":
                    pos.append(count)
                elif i=="N":
                    none.append(count)
                count+=1
            for i in pos:
                if i==0:
                    mv=random.sample(left,1)[0]
                elif i==len(self.river)-1:
                    mv=random.sample(right,1)[0]
                else:
                    mv=random.sample(move,1)[0]
                if self.river[i]=="F" and mv!=0:
                    if self.river[i+mv]=='N':
                        self.river[i]='N'
                        self.river[i+mv]='F'
                    elif self.river[i+mv]=="B":
                        self.river[i]="N"
                    else:
                        if len(none)>0:
                            rnd=random.sample(none,1)[0]
                            self.river[rnd]="F"
                            none.remove(pick)
                if self.river[i]=='B' and mv!=0:
                    if self.river[i+mv]!='B':
                        self.river[i]="N"
                        self.river[i+mv]="B"
                    else:
                        if len(none)>0:
                            rnd=random.sample(none,1)[0]
                            self.river[rnd]='B'
                            none.remove(rnd)
            tot+=1
            print(self.river)
            if self.ck(): return


def get(s):
    s0="Enter the "+s
    s1="Enter the right "+s
    print(s0,end=":")
    a=input()
    while True:
        if a.isdigit(): return int(a)
        print(s1,end=":")
        a=input()
def main():
    leng=get("length of river")
    fish=get("number of fish")
    bear=get("number of bear")
    river=list()
    pos=list(range(leng))
    for i in range(leng):
        river.append("N")
    fishlist=random.sample(pos,fish)
    for i in fishlist:
        pos.remove(i)
        river[i]="F"
    bearlist=random.sample(pos,bear)
    for i in bearlist:
        river[i]="B"
    a=ecosystem(fish,bear,river)
    print(a.river)
    step=get("number of step")
    a.solve(step)

main()