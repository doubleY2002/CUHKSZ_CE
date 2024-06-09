mk=[]
prime=[]
def ck(n):
    global mk
    s=str(n)
    s=s[::-1]
    w=eval(s)
    if mk[w]==0 and w!=n:
        return True
    else:
        return False
for i in range(0,10005):
    mk.append(0)
for i in range(2,10000):
    if mk[i]==0:
        prime.append(i)
        for j in range(2,int(10000/i)+1):
            mk[i*j]=1
i,cnt=0,0
while cnt<100:
    if ck(prime[i]):
        if cnt%10==9: print("%7d"%prime[i])
        else: print("%7d"%prime[i],end="")
        cnt+=1
    i+=1