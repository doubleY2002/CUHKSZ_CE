val=eval(input("Enter the final accont value:"))
rate=eval(input("Enter the annual interest rate:"))
year=eval(input("Enter the number of years:"))
init=val/(1+rate/100)**year
print("The initial value is: %lf"%init)