import numpy

def factorial_list(list):
    result = 1
    for i in list:
        result *= i
    return result


if __name__ == "__main__":
    a = [1,2,3,4,4]
    b = factorial_list(a[1:5])
    print(b)