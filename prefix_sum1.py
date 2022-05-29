'''
Find if there is a subarray with 0 sum

'''

def solutoin(arr, n):
    n_sum = 0
    s = set()

    for i in range(n):
        if n_sum = 0 or n_sum in s:
            return True
        s.add(n_sum)

    return False

arr = [1, 4, -2, -2, 1, 6]
print (solutoin(arr, len(arr)))
