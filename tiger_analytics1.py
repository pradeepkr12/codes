
class jar():
    def __init__(self, C, K):
        self.values = []
        self.totalcount = 0
        self.diff_threshold = K
        self.jar_limit = C
    def add_to_jar(self, x):
        if self.totalcount > self.jar_limit: return False
        for v in self.values:
            if abs(v-x) > self.diff_threshold:
                return False
        self.values.append(x)
        self.totalcount += 1
        return True

def process(jars, v, C, K):
    # insert in list of jars, if no jar found create a new jar
    value_added = False
    for jar_ in jars:
        if jar_.add_to_jar(v):
            value_added = True
            break

    if not value_added:
        newjar = jar(C, K)
        newjar.add_to_jar(v)
        jars.append(newjar)
    return jars

def solution(vals, C, K):
    jars = []
    for v in vals:
        jars = process(jars, v, C, K)
    print (len(jars))
    print ("Details for each jar")
    for jar_ in jars:
        print ("jar: ", jar_.values)

solution([1,2,3,4,6,12], 5, 6)
