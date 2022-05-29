def add_solution(s, wordlist, result):
    wordfound = False
    for word in wordlist:
        if word in s:
            wordfound = True
            break
    if wordfound:
        new_wordlist = wordlist[:]
        new_wordlist.remove(word)
        start = s.index(word)
        end = start + len(word)
        ns = s[:start]+s[end:]
        newresults = result[:]
        newresults.append(word)
        return add_solution(ns, new_wordlist, newresults), add_solution(s,
                                                                    new_wordlist,
                                                                    result)
    else:
        return ' '.join(result)
