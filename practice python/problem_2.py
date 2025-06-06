# letter='''Dear <|NAME|>,
# you are selected!
# date: <|DATE|>
# Thank you,'''
# name =input("emter the name: ")
# date=input("enter the date: ")
# print(letter.replace("<|NAME|>", name).replace("<|DATE|>", date))
# # print(letter)
lis= [1, 2, 3, 4, 5,5,7,3,8,9,0, 6, 7, 8, 9]
lis2= [1, 2, 3, 4, 5,5,3,9, 6, 7, 8, 9]
print(lis.pop(3))
print(lis.remove(5))
print(lis.extend([5,8,1]))
print(lis.count(5))
print(lis)
l=lis+lis2
print(l)
print(lis+lis2)
print(l.index(5))
print(l.sort())