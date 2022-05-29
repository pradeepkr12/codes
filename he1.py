'''
find all triplets such that 
i < j < k
a[i] < a[j] > a[k]

'''
from collections import defaultdict
def solve3(arr):
    """
    in this solution, I have to remember peaks that I've already seen
    """
    count = 0
    mags = defaultdict(int)
    for i in range(0,len(arr)-1):
        if arr[i]>=arr[i+1]:
            temp = 0
            for j in range(i+1, len(arr)):
                if arr[j]<arr[i]:
                    temp+=1
                if arr[j]<=arr[i]:
                    mags[j]-=1
            count+= (i+mags[i])*(temp)
    return count
def solution(a, n):
    c = 0
    for i in range(1, n-1):
        c1 = c2 = 0
        for j in range(0, i):
            if a[i] > a[j]:
                c1 += 1
        for j in range(i+1, n):
            if a[i] > a[j]:
                c2 += 1
        if c1 > 0 and c2 > 0:
            c += c1 * c2
    return c

def solution1(a):
    c = 0
    for i, vals in enumerate(a):
        if vals[i] > vals[i+1]:

[7, 5, 1, 6, 9, 2]

# n = input()
# vals = map(int, raw_input().split())
# n = 5
# a = [1, 2, 5, 4, 3]
# solution(a, n)

import random
all_test_cases = []
for _ in range(100):
    n = random.randint(10, 20)
    n = 6
    vals = [random.randint(1, 10) for _ in range(n)]
    s1 = solve3(vals)
    s2 = solution(vals, n)
    all_test_cases.append([vals, s1, s2, s1 == s2])

matches = 0
mismatched = []
for i, testcase in enumerate(all_test_cases):
    if testcase[-1]:
        matches += 1
    else:
        mismatched.append(i)
print ("Matches: ", matches)



