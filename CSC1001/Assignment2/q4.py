def isAnagram(s1,s2):
    S1=list(s1)
    S1.sort()
    S2=list(s2)
    S2.sort()
    if S1==S2:
        return True
    else:
        return False
s1=input("Please enter the first word:")
s2=input("Please enter the second word:")
if isAnagram(s1,s2):
    print("is an anagram")
else :
    print("is not an anagram")