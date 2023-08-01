import pickle

filename = "D:/Files/VisualStudioCode/TT2.0/Ubiquitous-Train/forTest/pickle.bin"

class Test:
    greeting = []


t = Test()
t.greeting = [1,23,3]
print('t.greeting', t.greeting[2]) 

with open(filename, 'wb+') as fp: 
    pickle.dump(t, fp)

with open(filename, 'rb') as fp:
    t2 = pickle.load(fp)

print('t2.greeting', t2.greeting)