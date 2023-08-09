from numpy import *

def b(a):
    a[2] = 9

if __name__ == '__main__':
    a = [2,3,[2],4,5,6]
    b(a)
    print(a)