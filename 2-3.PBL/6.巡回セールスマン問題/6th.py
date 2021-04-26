import random

# 5.2
import optimization

s = [4, 4, 4, 2, 2, 6, 6, 5, 5, 6, 6, 0]
optimization.printschedule(s)
l = [int(i) for i in range(12)]
random.shuffle(l)
print(l)
# optimization.printschedule2(l)

"""
#5.3
print( ' schedulecost = %s' %optimization.schedulecost(s))

#5.4 Random Search
domain = [(0,8)]*(len(optimization.people)*2)
print( ' ===== random optimization ======')
s = optimization.randomoptimize(domain,optimization.schedulecost)
print( s)
optimization.printschedule(s)
print( ' schedulecost = %s' %optimization.schedulecost(s))
#5.5 Hill Climb Search
domain = [(0,8)]*(len(optimization.people)*2)
print( ' ===== hill climb optimization ======')
s = optimization.hillclimb(domain,optimization.schedulecost)
print( s)
optimization.printschedule(s)
print( ' schedulecost = %s' %optimization.schedulecost(s))

#5.6 Simulated Annealing
domain = [(0,8)]*(len(optimization.people)*2)
print( ' ===== annealing optimization ======')
s = optimization.annealingoptimize(domain,optimization.schedulecost)
print( s)
optimization.printschedule(s)
print( ' schedulecost = %s' %optimization.schedulecost(s))
"""
countll = 0
# 5.7 Genetic Algorithm
print(' ===== genetic optimization ======')
print(l)
domain = [(0, 12)]*12
l = optimization.geneticoptimize(domain, optimization.schedulecost2)
print(l)
# optimization.printschedule(l)
print(' schedulecost = %s' % optimization.schedulecost2(l))

for abc in range(100):
    # 5.7 Genetic Algorithm
    #print( ' ===== genetic optimization ======')
    # print(l)
    domain = [(0, 12)]*12
    l = optimization.geneticoptimize(domain, optimization.schedulecost2)
    #print( l)
    #
    try:
        optimization.printschedule(l)
        print(' schedulecost = %s' % optimization.schedulecost2(l))
        print(l)
    except:
        pass
    #

    if(optimization.schedulecost2(l) == 11):
        countll += 1

print(countll)
