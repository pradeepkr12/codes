
def all_binary_search(arr, x): 
    low = 0
    high = len(arr) - 1
    mid = 0
    startidx = -1
    while low <= high: 
        mid = (high + low) // 2
        print ("Start", low, high, mid, startidx)

        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        elif arr[mid] == x:
            high = mid - 1
            startidx = mid
        print ("End", low, high, mid, startidx)

    print ("----")
    low = 0
    high = len(arr) - 1
    mid = 0
    endidx = -1
    while low <= high: 
        mid = (high + low) // 2
        print ("Start", low, high, mid, endidx)
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        elif arr[mid] == x:
            low = mid + 1
            endidx = mid
        print (low, high, mid, endidx)

    if (startidx != -1) and (endidx != -1):
        result = 0
        print (startidx, endidx)
        print (arr[startidx: endidx+1])
        for i in range(startidx, endidx+1):
            if arr[i] == x: result += 1
        return result
    return -1


all_binary_search([1, 2, 3, 3, 4], 3)
