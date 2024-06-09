def ck1(x):
    try:
        w=float(x)
        return w
    except:
        return 0

while True:
    m=input("Please enter a number: ")
    if ck1(m)>=1 and type(eval(m))==int:
        m=eval(m)
        break
    print("invalid input")

print("m\tm+1\tm**(m+1)")
for i in range(1,m+1):
    print("%d\t%d\t%d"%(i,i+1,i**(i+1)))