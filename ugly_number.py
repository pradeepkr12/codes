# https://www.geeksforgeeks.org/ugly-numbers/

def check_divisiblity(n, dp, m):
    '''
    divide number by n by m
    if divisible, check if it is present in dp
    if yes break, if not keep on dividing
    '''
    while (n%m == 0):
        nn = n//m
        if dp.get(nn) is not None:
            return True
        n = nn
    return n == 1

def solution(n):
    if n <= 3: return n
    dp = {}
    dp[1] = 1
    dp[2] = 1
    dp[3] = 1
    dp[5] = 1
    i = 0
    val = 1
    res = 0
    while i < n:
        if dp.get(val) is None:
            if check_divisiblity(val, dp, 2) or \
                    check_divisiblity(val, dp, 3) or \
                    check_divisiblity(val, dp, 5):
                dp[val] = 1
                res = val
                # print (res)
                i += 1
        elif dp[val]:
            i += 1
            res = val
            # print (res)
        # print (i, res, val, dp)
        val += 1
    return res

def check_divisibility(n, dp):
    possible_ug = [n]
    ug_flag = False
    while (n%2==0) or (n%3==0) or (n%5==0):
        if n%2 == 0:
            nn = n//2
        elif n%3 == 0:
            nn = n//3
        elif n%5 == 0:
            nn = n//5
        if dp.get(nn) is None:
            possible_ug.append(nn)
        else:
            ug_flag = True
            break
        n = nn
    if not ug_flag:
        return False
    if (n == 1) or ug_flag:
        for v in possible_ug: dp[v]=1
        return True
    else:
        return False

def solution(n):
    if n <=3:  return n
    dp = {}
    dp[1] = 1
    dp[2] = 1
    dp[3] = 1
    dp[5] = 1
    i = 0
    val = 1
    res = 0
    while i < n:
        if dp.get(val) is None:
            if check_divisibility(val, dp):
                dp[val] = 1
                res = val
                # print (res)
                i += 1
        elif dp[val]:
            i += 1
            res = val
            # print (res)
        # print (i, res, val, dp)
        val += 1
    return res

def solution(n):
    DP = [0] * N
    dp[0] = 1
    i2 = i3 = i5 = 0
    next2, next3, next5 = 2, 3, 5
    for i in range(1, n):
        dp[i] = min(next2, next3, next5)
        if dp[i] == next2:
            i2 += 1
            next2 = dp[i2] * 2
        if dp[i] == next3:
            i3 += 1
            next3 = dp[i3] * 3
        if dp[i] == next5:
            i5 += 1
            next5 = dp[i5] * 5
        # print (i, i2, i3, i5, dp)
    return dp[-1]
