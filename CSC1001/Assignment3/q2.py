class polynomial:
    def __init__(self,xs=0,m=0,var="x"):
        self.xs=xs
        self.m=m
        self.var=var
    def derivative(self):
        derivative=polynomial()
        derivative.xs=self.xs*self.m
        derivative.m=self.m-1
        derivative.var=self.var
        return derivative
    def get(self,s,f):
        xs=f
        m=0
        num=""
        f1=False
        f2=False
        var=""
        for i in s:
            if i=="^":
                f1=True
            elif i=="*":
                if f1==True and f2==False:
                    m=eval(num)
                    num=""
                    f2=True
                else:
                    xs=xs*eval(num)
                    num=""
            elif i.isdigit():num+=i
            else: var=i
        if num!="":
            if f1==True: m=eval(num)
            else:xs=xs*eval(num)
        if var!="" and m==0: m=1
        self.__init__(xs,m,var)
    def print(self,f=True):
        if self.xs==0: return
        if f==True and self.xs>0: print(end="+")
        if self.m==0:
            print(self.xs,end="")
            return
        if self.xs==-1: print(end="-")
        elif self.xs!=1 or self.m==0: print(self.xs,end="*")
        if self.m==1: print(self.var,end="")
        elif self.m!=0:print("%s^%d"%(self.var,self.m),end="")

def main():
    polynom=input("Give me a polynomial:")
    a=list()
    la=0
    f=1
    here=polynomial()
    for i in range(len(polynom)):
        nw=polynom[i]
        if nw=="+" or nw=='-':
            here.get(polynom[la:i],f)
            if la==0:
                (here.derivative()).print(False)
            else:
                (here.derivative()).print()
            la=i+1
            if nw=="+":f=1
            else: f=-1
    here.get(polynom[la:],f)
    (here.derivative()).print()
main()