import math

def sol(a):
    n = len(a)
    min_jumps = [math.inf] * n
    min_jumps[0] = 0
    for i in range(0, n):
        print (f"Ranges from {i} - {min(i+a[i]+1, n)}")
        for j in range(i+1, min(i+a[i]+1, n)):
            min_jumps[j] = min(min_jumps[j], 1 + min_jumps[i])
            print (i, j, min_jumps)
    return min_jumps[-1]

def greedy_sol(a):
    n = len(a)
    min_jumps = [math.inf] * n
    min_jumps[0] = 0
    greedy = 0
    counter = 0
    greedy_counter = 0
    for i in range(0, n):
        for j in range(i+1, min(i+a[i]+1, n)):
            counter += 1
            min_jumps[j] = min(min_jumps[j], 1 + min_jumps[i])
            if min_jumps[j] + a[j] >= n:
                greedy= 1 + min_jumps[j]
                greedy_counter = counter
    return min_jumps[-1], counter, greedy, greedy_counter
# a = [2, 3, 1, 1, 4]
# print (sol(a))

def jump(a):
    previous = 0
    current = 0
    jumps = 0
    for i in range(len(a)):
        if i > previous:
            jumps += 1
            previous = current
        current = max(current, i + a[i])
        print (i, previous, current, jumps)
    return jumps


all_data = []
solutions = []
for _ in range(100):
    a_len = np.random.randint(100)
    a = [np.random.randint(10) for __ in range(10)]
    result = greedy_sol(a)
    all_data.append(a)
    solutions.append(result)

# solutions which are greedy
greedysol = 0
for i, s in enumerate(solutions):
    if s[1] < s[3]:
        greedysol += 1
    if s[0] == s[2]:
        print (i, s[1], s[3])
