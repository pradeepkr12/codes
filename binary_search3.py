X = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
X = [2, 10, 6, 14, 1]
N = len(X)
P = [0] * N
M = [0] * (N + 1)

L = 0
for i in range(N-1):
    # // Binary search for the largest positive j ≤ L
    # // such that X[M[j]] <= X[i]
    lo = 1
    hi = L
    while lo <= hi:
        mid = (lo + hi)//2
        if X[M[mid]] < X[i]:
            lo = mid+1
        else:
            hi = mid-1

    # // After searching, lo is 1 greater than the
    # // length of the longest prefix of X[i]
    newL = lo

    # // The predecessor of X[i] is the last index of 
    # // the subsequence of length newL-1
    P[i] = M[newL-1]
    M[newL] = i
    if newL > L:
        # // If we found a subsequence longer than any we've
        # // found yet, update L
        L = newL

# // Reconstruct the longest increasing subsequence
S = [0] * (L+1)
k = M[L]
for i in range(L-1, -1, -1):
    S[i] = X[k]
    k = P[k]

