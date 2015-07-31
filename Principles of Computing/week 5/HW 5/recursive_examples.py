def add_up(n):
    if n == 0:
        return 0
    else:
        print n
        return n + add_up(n - 1)

#print add_up(4)
print add_up(10)
print "#############"

def multiply_up(n):
    if n == 0:
        return 1
    else:
        print n
        return n * multiply_up(n - 1)
    
print multiply_up(6)
print "#############"

def fib(num):
    print num
    if num == 0:
        return 0
    elif num == 1:
        return 1
    else:
        return fib(num - 1) + fib(num - 2)

counter = 0   
print fib(4)
print "#############"
def memoized_fib(num, memo_dict):
    print num
    global counter
    counter += 1
    if num in memo_dict:
        return memo_dict[num]
    else:
        sum1 = memoized_fib(num - 1, memo_dict)
        sum2 = memoized_fib(num - 2, memo_dict)
        memo_dict[num] = sum1 + sum2
        return sum1 + sum2

print memoized_fib(11, {0 : 0, 1 : 1})
print "counter ", counter