import time
import random
import math
# mutate corossover pop
people = [('Seymour', 'BOS'),
          ('Franny', 'DAL'),
          ('Zooey', 'CAK'),
          ('Walt', 'MIA'),
          ('Buddy', 'ORD'),
          ('Les', 'OMA')]
# Laguardia
destination = 'LGA'

flights = {}
#
fp = open('./schedule.txt')
lines = fp.read().strip().split('\n')  # Arranged by A.Fujii 2017/Sep
for line in lines:
    origin, dest, depart, arrive, price = line.split(',')
    flights.setdefault((origin, dest), [])

    # Add details to the list of possible flights
    flights[(origin, dest)].append((depart, arrive, int(price)))


def getminutes(t):
    x = time.strptime(t, '%H:%M')
    return x[3]*60+x[4]


def printschedule(r):
    for d in range(int(len(r)/2.0)):  # Arranged for Python3.4
        name = people[d][0]
        origin = people[d][1]
        out = flights[(origin, destination)][int(r[d])]
        ret = flights[(destination, origin)][int(r[d+1])]
        print('%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name, origin,
                                                      out[0], out[1], out[2],
                                                      ret[0], ret[1], ret[2]))


def printschedule2(r):  # 改良したもの
    print(r)


def schedulecost(sol):
    totalprice = 0
    latestarrival = 0
    earliestdep = 24*60

    for d in range(int(len(sol)/2.0)):  # Arranged for python3.4 by A.Fujii
        # Get the inbound and outbound flights
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[d])]
        returnf = flights[(destination, origin)][int(sol[d+1])]

        # Total price is the price of all outbound and return flights
        totalprice += outbound[2]
        totalprice += returnf[2]

        # Track the latest arrival and earliest departure
        if latestarrival < getminutes(outbound[1]):
            latestarrival = getminutes(outbound[1])
        if earliestdep > getminutes(returnf[0]):
            earliestdep = getminutes(returnf[0])

    # Every person must wait at the airport until the latest person arrives.
    # They also must arrive at the same time and wait for their flights.
    totalwait = 0
    for d in range(int(len(sol)/2.0)):  # Arranged for python3.4 by A.Fujii
        origin = people[d][1]
        outbound = flights[(origin, destination)][int(sol[d])]
        returnf = flights[(destination, origin)][int(sol[d+1])]
        totalwait += latestarrival-getminutes(outbound[1])
        totalwait += getminutes(returnf[0])-earliestdep

    # Does this solution require an extra day of car rental? That'll be $50!
    if latestarrival > earliestdep:
        totalprice += 50

    return totalprice+totalwait


def schedulecost2(sol):  # 改良したもの
    Sum = 0
    for i in range(0, len(sol)-1):
        Sum += abs(sol[i]-sol[i+1])

    return Sum


def randomoptimize(domain, costf):
    best = 999999999
    bestr = None
    for i in range(0, 1000):
        # Create a random solution
        r = [float(random.randint(domain[i][0], domain[i][1]))
             for i in range(len(domain))]

        # Get the cost
        cost = costf(r)

        # Compare it to the best one so far
        if cost < best:
            best = cost
            bestr = r
    return r


def hillclimb(domain, costf):
    # Create a random solution
    sol = [random.randint(domain[i][0], domain[i][1])
           for i in range(len(domain))]
    # Main loop
    while 1:
        # Create list of neighboring solutions
        neighbors = []

        for j in range(len(domain)):
            # One away in each direction
            if sol[j] > domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])
            if sol[j] < domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])

        # See what the best solution amongst the neighbors is
        current = costf(sol)
        best = current
        for j in range(len(neighbors)):
            cost = costf(neighbors[j])
            if cost < best:
                best = cost
                sol = neighbors[j]

        # If there's no improvement, then we've reached the top
        if best == current:
            break
    return sol


def annealingoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    # Initialize the values randomly
    vec = [float(random.randint(domain[i][0], domain[i][1]))
           for i in range(len(domain))]

    while T > 0.1:
        # Choose one of the indices
        i = random.randint(0, len(domain)-1)

        # Choose a direction to change it
        dir = random.randint(-step, step)

        # Create a new list with one of the values changed
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        # Calculate the current cost and the new cost
        ea = costf(vec)
        eb = costf(vecb)
        p = pow(math.e, (-eb-ea)/T)

        # Is it better, or does it make the probability
        # cutoff?
        if (eb < ea or random.random() < p):
            vec = vecb

        # Decrease the temperature
        T = T*cool
    return vec


def geneticoptimize(domain, costf, popsize=50, step=1,
                    mutprob=0.2, elite=0.2, maxiter=100):
    def mutate(vec):
        i = random.randint(0, len(domain)-1)
        j = random.randint(0, len(domain)-1)
        vec[i], vec[j] = vec[j], vec[i]
        return vec

    # Crossover Operation
    def crossover(vec):  # 違う配列同士をくっつける
        i = random.randint(1, len(domain)-2)
        crossover_vec = vec[i:]
        random.shuffle(crossover_vec)
        res = vec[0:i] + crossover_vec

        return res

    # Build the initial population
    pop = []
    for i in range(popsize):
        vec = [i for i in range(0, 12)]
        random.shuffle(vec)
        pop.append(vec)

    # How many winners from each generation?
    topelite = int(elite*popsize)  # スケジュールコストが上位20%のみ選出

    # Main loop
    for i in range(maxiter):
        scores = [(costf(v), v) for v in pop]  # costfでスケジュールコストを計算
        scores.sort()  # costfのスコア順でソート
        ranked = [v for (s, v) in scores]  # (cost,[遺伝子列1]) s,スコア　v,遺伝子列
        # Start with the pure winers
        pop = ranked[0:topelite]  # popの中にtop20が入る

        # Add mutated and bred forms of the winners
        while len(pop) < popsize:
            if random.random() < mutprob:
                # Mutation　突然変異
                c = random.randint(0, topelite)
                pop.append(mutate(ranked[c]))
            else:
                # Crossover　交差
                c = random.randint(0, topelite)
                pop.append(crossover(ranked[c]))  # 一次元に直した
            # Print current best score
        #print( scores[0][0])
    return scores[0][1]
