


def solution(a):
    n = len(a)
    dp1 = [0] * n
    dp2 = [0] * n
    for i, val in enumerate(a):
        val1 = i + val
        val2 = i - val
        if 0 < val1 < n:
            dp1[i] = val1
        if 0 < val2 < n:
            dp2[i] = val2
    start = n
    reached = False
    can_reach(a, start)

def can_reach(a, n):
    print (f"Now at {n}, {a[n]}")
    if a[n] == 0:
        return True
    val1 = n + a[n]
    val2 = n - a[n]
    res1 = res2 = False
    if 0 < val1 < len(a):
        print ("Taking step1")
        res1 = can_reach(a, val1)
    if 0 < val2 < len(a):
        print ("Taking step2")
        res2 = can_reach(a, val2)
    if res1 or res2:
        return True
    return False


def can_reach2(a, n, dp):
    print (f"Now at {n}, {a[n]}", dp)
    if dp[n]:
        return 1
    val1 = n + a[n]
    val2 = n - a[n]
    res1 = res2 = False
    if 0 < val1 < len(a):
        print ("Taking step1")
        dp[val1] = can_reach2(a, val1, dp)
    if 0 < val2 < len(a):
        print ("Taking step2")
        dp[val2] = can_reach2(a, val2, dp)
    return dp[n]

def can_reach3(a, n, dp, visited):
    if 0 <= n < len(a):
        if visited[n] == 1:
            if dp[n] or (a[n] == 0):
                return 1
            else:
                return 0
        visited[n] = 1
        val1 = n + a[n]
        val2 = n - a[n]
        print (f"Now at {n}, {a[n]}, {val1}, {val2}", dp)
        dp[n] = can_reach3(a, val1, dp, visited) or can_reach3(a, val2, dp,
                                                                visited)
        return dp[n]
    else:
        return 0


def solve(a, n):
    visited = [0] * len(a)
    dp = [0] * len(a)
    can_reach3(a, n, dp, visited)

for test_case, n in zip([[4,2,3,0,3,1,2], [4,2,3,0,3,1,2], [3,0,2,1,2]],
                        [5, 0, 2]):
    print ("For test case: ", test_case, n, " Result: ", solve(test_case, n))
