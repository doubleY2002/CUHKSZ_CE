mk=list()
for i in range(0,101):
    mk.append(0)
for i in range(1,101):
    for j in range(1,100//i+1):
        mk[i*j]=1-mk[i*j]
for i in range(1,101):
    if mk[i]:
        print(i,end=" ")