
def bsearch(a, x):
    low = 0
    high = len(a)
    while low <= high:
        mid = (low + high)//2
        if a[mid] > x:
            high = mid - 1
        elif a[mid] < x:
            low = mid + 1
        else:
            return True
    return False

