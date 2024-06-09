a=[0,31,28,31,30,31,30,31,31,30,31,30,31]
day=eval(input("input day:"))
month=eval(input("input month:"))
if(a[month]==day):
    month=month%12+1
    day=1
else:
    day=day+1
print("month="+str(month));
print("day="+str(day));