def isValid(number):
    ans=sumOfDoubleEvenPlace(number)+sumOfOddPlace(number)
    if ans%10==0:
        return True
    else:
        return False
def sumOfDoubleEvenPlace(number):
    cnt=0
    f=1
    while number:
        x=number%10
        number//=10
        f=1-f
        if f==1:
            cnt+=getDigit(x)
    return cnt
def getDigit(number):
    number*=2
    if number>9:
        number=number%10+number//10
    return number
def sumOfOddPlace(number):
    cnt=0
    f=0
    while number:
        x=number%10
        number//=10
        f=1-f
        if f==1:cnt+=x
    return cnt
Card=eval(input("Please enter your credit card numbers:"))
if isValid(Card):
    print("It is valid.")
else :
    print("It is invalid.")