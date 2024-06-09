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
cnt=0
print("The prime numbers smaller than %d include:"%m)
for x in range (2,m):
    mk=1
    for y in range (2,x):
        if x%y ==0 :
            mk=0
    if mk==1:
        print(x,end=" ")
        cnt=cnt+1
        if cnt%8==0:
            print("")