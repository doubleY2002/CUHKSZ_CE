def getnext(lastGuess,n):
    nextGuess=1.0*(lastGuess+(n/lastGuess))/2
    if abs(nextGuess-lastGuess)<0.0001:
        return nextGuess
    else:
        return getnext(nextGuess,n)
lastGuess=eval(input("Please enter a number:"))
while lastGuess<0:
    lastGuess=eval(input("Please enter a positive number:"))
print("Here is the square root:",getnext(1,lastGuess))