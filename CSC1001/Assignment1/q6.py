def ck1(x):
    try:
        w=float(x)
        return w
    except:
        return 0

def ck2(x):
    try:
        w=float(x)
        return True
    except:
        return False

def read(x):
    while True:
        m=input("Please enter a number %s: "%x)
        if ck2(m):
            return eval(m)
        print("invalid input")

fu=input("Please enter a f: ")
while fu!="sin" and fu!="cos" and fu!="tan":
    print("invaild function")
    fu=input("Please enter a right f: ")

a=read("a")
b=read("b")
while True:
    n=input("Please enter a number n: ")
    if ck1(n)>=1 and type(eval(n))==int:
        n=eval(n)
        break
    print("invalid input")
ans=0.0
from math import sin
from math import cos
from math import tan
if fu=="sin":
    for i in range(n):
        ans=ans+sin(1.0*a+1.0*(b-a)/n*(i-0.5))
if fu=="cos":
    for i in range(n):
        ans=ans+cos(1.0*a+1.0*(b-a)/n*(i-0.5))
if fu=="tan":
    for i in range(n):
        ans=ans+tan(1.0*a+1.0*(b-a)/n*(i-0.5))
print(ans*(b-a)/n)