#!/usr/bin/python
import sys
pDict = {}
mDict = {}
pCount = 0
mCount = 0

f = open(sys.argv[1], 'r')
for line in f:
    line = line.strip()
    [pIdx, mIdx, val] = line.split(",")
    if pIdx not in pDict:
        pDict[pIdx] = pCount
        pCount += 1
    if mIdx not in mDict:
        mDict[mIdx] = mCount
        mCount += 1
    print "%d,%d,%s" %(pDict[pIdx], mDict[mIdx], val)
f.close()
