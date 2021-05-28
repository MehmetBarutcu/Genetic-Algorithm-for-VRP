import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import operator
import itertools

class GeneticAlgorithm:

    def __init__(self, distanceMatrix, popSize, eliteSize, mutationRate,tmutateRate, generations,kmCost, tCapacity, aCost, demand,
                 encode, empty,ftime,vtime,ttype,tallowed,tmutateFlip, nTrucks=4, infCost=9999, uCost=14, tSpeed=50, ovCost=60, factor=1000,initial=[]):

        self.distanceMatrix = distanceMatrix
        self.popSize = popSize
        self.eliteSize = eliteSize
        self.mutationRate = mutationRate
        self.generations = generations
        self.kmCost = kmCost
        self.tCapacity = tCapacity
        self.aCost = aCost
        self.infCost = infCost
        self.uCost = uCost
        self.tSpeed = tSpeed
        self.ovCost = ovCost
        self.factor = factor
        self.demand = demand
        self.encode = encode
        self.empty = empty
        self.nTrucks = nTrucks
        self.bestSol = None
        self.initial = initial
        self.ftime = ftime
        self.vtime = vtime
        self.ttype = ttype
        self.tallowed = tallowed
        self.tmutaterate = tmutateRate
        self.tmutatereverse = tmutateFlip

    def __str__(self):
        return f'Genetic Algorithm object with {self.get_customer_number()} customers'

    def get_distance_matrix(self):
        return self.distanceMatrix

    def get_popsize(self):
        return self.popSize

    def get_elitesize(self):
        return self.eliteSize

    def get_mutationrate(self):
        return self.mutationRate

    def get_generations(self):
        return self.generations

    def get_kmcost(self):
        return self.kmCost

    def get_truckcapacities(self):
        return self.tCapacity

    def get_truckactivecost(self):
        return self.aCost

    def get_truckspeed(self):
        return self.tSpeed

    def get_overtimecost(self):
        return self.ovCost

    def get_customerdemands(self):
        return self.demand

    def get_encodedsolution(self):
        return self.encode

    def get_emptycylinders(self):
        return self.empty

    def get_numberoftrucks(self):
        return self.nTrucks

    def get_customer_number(self):
        return len(self.distanceMatrix)


    def check_feasibility(self):
        gen = self.breakRoutes(self.bestSol)
        loads = self.initialLoads(gen)
        times = self.calculate_collecttime(self.bestSol)
        #print('Times:',times)
        trucks = [f'T{i+1}' for i in range(len(gen))]
        trucks.append('Total')
        status = pd.DataFrame(index=trucks,columns=['Km','Loaded Cylinders','Capacity','Working Time','Working Hour','Truck Type'])
        tkm = self.kmcost(self.bestSol, self.distanceMatrix, self.encode)
        status.loc['Total']['Km'] = sum(tkm)
        for i,g in enumerate(gen):
            ttype = 0
            for j in g:
                ttype +=self.check_trucktype(i,j)
                if ttype >= self.infCost:
                    status.iloc[i]['Truck Type'] = 'Infeasible'
                    break
                else:
                    status.iloc[i]['Truck Type'] = 'Feasible'
            status.iloc[i]['Loaded Cylinders'] = loads[i][0]
            status.iloc[i]['Km'] = tkm[i]
            if loads[i][0] >= self.tCapacity[i]:
                status.iloc[i]['Capacity'] = 'Infeasible'
            else:
                status.iloc[i]['Capacity'] = 'Feasible'
            if (((tkm[i]-(self.return_cost(i,g)/self.kmCost[i]))/ self.tSpeed) + times[i]) > 9:
                status.iloc[i]['Working Time'] = 'Infeasible'
                status.iloc[i]['Working Hour'] = ((tkm[i] / self.tSpeed) + times[i])
                #print(self.return_cost(i,g)/self.kmCost[i])
            elif ((tkm[i] / self.tSpeed) + times[i]) > 9:
                #print((tkm[i]-(self.return_cost(i,g)/self.kmCost[i]))/ self.tSpeed)
                status.iloc[i]['Working Time'] = 'Overtime'
                status.iloc[i]['Working Hour'] = ((tkm[i]/self.tSpeed) + times[i])
            else:
                status.iloc[i]['Working Time'] = 'Feasible'
                status.iloc[i]['Working Hour'] = (tkm[i] / self.tSpeed) +times[i]
        status = status.fillna(value='-')
        return status

    def check_collectedcylinders(self):
        routes = self.breakRoutes(self.bestSol)
        loads = self.initialLoads([self.bestSol])
        for i,route in enumerate(routes):
            route = [i for i in route if i !=0]
            if len(route)>0:
                capacity = self.initial_cylinders(i,0,loads)
                index = ['Actual',f'T{i+1}_Collected','Difference']
                df = pd.DataFrame(index=index,columns=route)
                for j in route:
                    capacity += self.demand[self.encode[j]]-self.empty[self.encode[j]]
                    df.loc[f'T{i+1}_Collected'][j] = self.empty[self.encode[j]] if capacity >0 else self.empty[self.encode[j]] -abs(capacity)
                    df.loc['Actual'][j] = self.empty[self.encode[j]]
                    df.loc['Difference'][j]=df.loc['Actual'][j]-df.loc[f'T{i+1}_Collected'][j]
                    capacity = max(0,capacity)
                print('------------------------------------------')
                print(df)
                print('------------------------------------------')

    def calculate_collecttime(self,gen):
        times = []
        gen = self.breakRoutes(gen)
        for g in gen:
            t = 0
            for j in g:
                t += self.ftime[j] + (self.vtime[j]*(self.empty[self.encode[j]]+self.demand[self.encode[j]]))
            times.append(t/60)
        return times


    def initial_population(self):
        if len(self.initial) !=0 :
            customers = self.initial
            self.nTrucks = 1
        else:
            nCustomers = self.get_customer_number()
            customers = np.arange(1, nCustomers)

        comb_list = []
        #if len(self.initial) > 0:
            #comb_list.append(self.initial)
        for i in range(self.popSize):
            comb_list.append(np.random.permutation(customers))
            comb_list[i] = np.append(comb_list[i], 0)
            comb_list[i] = np.insert(comb_list[i], 0, 0)
            for _ in range(self.nTrucks - 1):
                comb_list[i] = np.insert(comb_list[i], random.randint(0, len(comb_list[i])), 0)
        #print(comb_list)
        #print(len(comb_list))
        return comb_list

    @classmethod
    def uniqueSets(cls,arr):
        return np.unique(arr)

    @classmethod
    def seperateDepots(cls,arr):
        return arr[arr!=0]

    def convertlisttonumpy(self,l):
        return np.array(l)

    @classmethod
    def breakRoutes(cls,individual):
        b = np.argwhere(individual == 0)
        routes = []
        for i in range(len(b) - 1):
            route = []
            for j in range(b[i][0], b[i + 1][0] + 1):
                route.append(individual[j])
            routes.append(route)
        return routes

    def initialLoads(self,gen):
        demands = []
        demand = 0
        for g in gen:
            dg = []
            for i in range(1, len(g)):
                if g[i] == 0:
                    dg.append(demand)
                    demand = 0
                demand += self.demand[self.encode[g[i]]]
            demands.append(dg)
        return demands

    def fitnessFunc(self,gen,loads):
        route_dict = {}
        for routeIndex, g in enumerate(gen):
            route_dict[routeIndex] = self.factor / self.calculate_chromosome_cost(g,loads,routeIndex)
        #print(route_dict)
        #print(100/route_dict[0])
        return sorted(route_dict.items(), key=operator.itemgetter(1), reverse=True)

    @classmethod
    def return_costt(cls,route,distanceMatrix,encode):
        return distanceMatrix.loc[encode[route[-2]]][encode[route[-1]]]

    @classmethod
    def kmcost(cls,gen,distanceMatrix,encode):
        km_list = []
        gen = cls.breakRoutes(gen)
        for g in gen:
            km = 0
            for i in range(len(g)-1):
                km += distanceMatrix.loc[encode[g[i]]][encode[g[i+1]]]
            km_list.append(km)
        return km_list

    def check_trucktype(self,index1,index2):
        #print(index1)
        #print(index2)
        #print(self.tallowed[index2])
        return self.infCost if self.ttype[index1] not in self.tallowed[index2] else 0

    def calculate_chromosome_cost(self, gen, loads, routeIndex):
        routes = self.breakRoutes(gen)
        t = self.calculate_collecttime(gen)
        #print(t)
        sCost = 0
        typecost = 0
        for index, route in enumerate(routes):
            kmCost = self.kmcost(np.array(route),self.distanceMatrix,self.encode)[0]*self.kmCost[index]
            #print(f'km cost: {kmCost}')
            rCost = self.return_cost(index, route)
            #print(f'return cost: {rCost}')
            cCost = self.cylinder_cost(index, routeIndex, loads,route)
            #print(f'cylinder: {cCost}')
            cCapacity = self.initial_cylinders(index, routeIndex, loads)
            #print(f'capacity: {cCapacity}')
            #if cCapacity >= self.infCost:
                #print(f'Infeasible Due to overload')
            for j in range(len(route) - 1):
                cCapacity += self.calculate_net_cylinders(route, j)
                #if cCapacity < 0:
                    #print(f'Uncollected cylinders at customer j')
                cCost += self.capacity_cost(cCapacity)
                #print(f'cCost {cCost}')
                cCapacity = self.check_uncollected_capacity(cCapacity)
                typecost += self.check_trucktype(index,route[j])
                #print(f'truck {index}')
                #print(f'typecost {typecost}')
            sCost += self.calculate_total_cost(cCost, kmCost, rCost, index,t) + typecost
            #print(f'sCost {sCost}')
        return sCost

    def return_cost(self,i,route):
        return self.kmCost[i] * self.distanceMatrix.loc[self.encode[route[-2]]][self.encode[route[-1]]]

    def usage_cost(self,i,r):
        return self.aCost[i] if len(r)>2 else 0

    def check_capacity(self,i,route,loads):
        #if self.tCapacity[i] < loads[route][i]:
            #print(f'Infeasible Due to overload')
        #print(loads[route])
        return self.infCost if self.tCapacity[i] < loads[route][i] else 0

    def initial_cylinders(self,i,route,loads):
        return max(0, self.tCapacity[i] - loads[route][i])

    def cylinder_cost(self,i,route,loads,r):
        return self.usage_cost(i,r) + self.check_capacity(i,route,loads)

    def calculate_net_cylinders(self,route,index):
        return self.demand[self.encode[route[index]]] - self.empty[self.encode[route[index]]]

    def check_uncollected_capacity(self,capacity):
        return 0 if capacity <0 else capacity

    def capacity_cost(self,capacity):
        return self.uCost * abs(capacity) if capacity <0 else 0

    def km_cost(self,encode,route,index1,index2):
        return self.kmCost[index1] * self.distanceMatrix.loc[encode[route[index2]]][encode[route[index2+1]]]

    def calculate_total_cost(self,cCost,kmCostt,rCost,index,time):
        if (((kmCostt)-rCost) / (self.tSpeed * self.kmCost[index]))+time[index] > 9:
            #print(time)
            #print(self.kmCost[index])
            #print(((kmCostt)-rCost) / (self.tSpeed * self.kmCost[index]))
            #print('Infeasible due to overtime')
            return self.infCost + cCost + kmCostt + (((kmCostt) / (self.tSpeed*self.kmCost[index]) + time[index] - 9) * self.ovCost)
        elif ((kmCostt) / (self.tSpeed * self.kmCost[index])) + time[index] > 9:
            #print(time[index])
            #print('Overtime')
            return ((((kmCostt) / (self.tSpeed*self.kmCost[index])) + time[index] - 9) * self.ovCost) + cCost + kmCostt
        else:
            #print('Feasible')
            return kmCostt + cCost

    def selection(self,popRanked):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
        for i in range(0, self.eliteSize):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - self.eliteSize):
            pick = 100 * random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults

    def matingPool(self,population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool

    def breed(self,parent1, parent2):
        p1 = parent1[1:-1]
        p2 = parent2[1:-1]
        child = []
        childP1 = []
        childP2 = []
        geneA = random.randint(0, len(p1) - 1)
        geneB = random.randint(0, len(p1) - 1)

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(p1[i])
        count = childP1.count(0)

        for item in p2:
            if item == 0:
                if self.nTrucks - count - 1 > 0:
                    childP2.append(item)
                    count += 1
            else:
                if item not in childP1:
                    childP2.append(item)
        child = childP1 + childP2
        child = np.append(child, 0)
        child = np.insert(child, 0, 0)
        return child

    def breedPopulation(self,matingpool):
        children = []
        length = len(matingpool) - self.eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, self.eliteSize):
            children.append(matingpool[i])

        for i in range(0, length):
            child = self.breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children


    def mutateTrucks(self,individual):
        #print(f'Original \n {individual}')
        individual = np.array(individual)
        individual = self.breakRoutes(individual)
        for i in individual:
            i = i.remove(0)
        individual = np.array(individual, dtype=object)
        for swapped in range(len(individual)):
            if (random.random() < self.tmutaterate):
                possibleTrucks = [i for i in range(len(individual)) if self.ttype[swapped] != self.ttype[i]]
                swapWith = random.choice(possibleTrucks)
                city1 = individual[swapped]
                city2 = individual[swapWith]
                individual[swapped] = city2
                individual[swapWith] = city1
                break
        individual = list(itertools.chain(*individual))
        individual.insert(0, 0)
        individual = np.array(individual)
        #print(f'mutated \n {individual}')
        return individual


    def mutateInverse(self,individual):
        individual = self.breakRoutes(individual)
        for i in individual:
            i = i.remove(0)
        for swapped in range(len(individual)):
            #print(individual)
            if (random.random() < self.tmutatereverse):
                individual[swapped].pop()
                individual[swapped].reverse()
                individual[swapped].append(0)
                break
        individual = list(itertools.chain(*individual))
        individual.insert(0,0)
        return individual

    def mutateInversePopulation(self,population):
        mutatedPop = []
        for i in range(self.eliteSize):
            mut = population[i].copy()
            m = self.mutateInverse(mut)
            loadsm = self.initialLoads([m])
            load = self.initialLoads([population[i]])
            print(m)
            if self.fitnessFunc([m], loadsm)[0][1] > self.fitnessFunc([population[i]], load)[0][1]:
                print(self.fitnessFunc([np.array(m)], loadsm)[0][1])
                mutatedPop.append(m)
            else:
                mutatedPop.append(population[i])
            for ind in range(self.eliteSize, len(population)):
                mutatedInd = self.mutateTrucks(population[ind])
                mutatedPop.append(mutatedInd)
        return mutatedPop

    def mutateTrucksPopulation(self,population):
        mutatedPop = []
        for i in range(self.eliteSize):
            mut = population[i].copy()
            m = self.mutateTrucks(mut)
            loadsm = self.initialLoads([m])
            load = self.initialLoads([population[i]])
            if self.fitnessFunc([m], loadsm)[0][1] > self.fitnessFunc([population[i]], load)[0][1]:
                #print(loadsm)
                #print(load)
                #print(m)
                #print(population[i])
                mutatedPop.append(m)
            else:
                #print(loadsm)
                #print(load)
                #print(m)
                #print(self.fitnessFunc([m], loadsm)[0][1])
                #print(self.fitnessFunc([population[i]], load)[0][1])
                mutatedPop.append(population[i])
        for ind in range(self.eliteSize, len(population)):
            mutatedInd = self.mutateTrucks(population[ind])
            #print(population[ind])
            #print(mutatedInd)
            mutatedPop.append(mutatedInd)
        return mutatedPop

    def mutate(self,individual):
        ind = individual[1:-1]
        for swapped in range(len(ind)):
            if (random.random() < self.mutationRate):
                swapWith = int(random.random() * len(ind))

                city1 = ind[swapped]
                city2 = ind[swapWith]

                ind[swapped] = city2
                ind[swapWith] = city1
        ind = np.append(ind, 0)
        ind = np.insert(ind, 0, 0)
        #print(ind)
        return ind

    def mutatePopulation(self,population):
        mutatedPop = []
        for i in range(self.eliteSize):
            mut = population[i].copy()
            m = self.mutate(mut)
            loadsm = self.initialLoads([m])
            load = self.initialLoads([population[i]])
            #print(type(population[i]))
            #print(type(m))
            #print(load)
            #print(loadsm)
            if self.fitnessFunc([np.array(m)],loadsm)[0][1] > self.fitnessFunc([population[i]],load)[0][1]:
                mutatedPop.append(m)
            else:
                mutatedPop.append(population[i])
        for ind in range(self.eliteSize, len(population)):
            mutatedInd = self.mutate(population[ind])
            mutatedPop.append(mutatedInd)
        return mutatedPop

    def concatTrucks(self,individual):
        concatWith = 0
        #print('original',individual)
        individual = np.array(individual)
        individual = self.breakRoutes(individual)
        indices = [j for j,i in enumerate(individual) if len(i) >2]
        #print(indices)
        df = pd.DataFrame(index=indices,columns=["Lenght"])
        for j,i in enumerate(individual):
            if len(i) > 2:
                df.loc[j]['Lenght'] = 1/len(i)
            i.pop(-1)
        df['cum_sum'] = df.Lenght.cumsum()
        df['cum_perc'] = 100 * df.cum_sum/ df.cum_sum.max()
        #print(df)
        individual = np.array(individual, dtype=object)
        for concat in range(len(individual)):
            if (random.random() < self.tmutaterate):
                pick = 100*random.random()
                for i in indices:
                    if pick <= df.loc[i]['cum_perc']:
                        concatWith = i
                        break
                if concat != concatWith:
                    individual[concat].extend(cust for cust in individual[concatWith] if cust not in individual[concat])
                    #individual[concat] = list(set(individual[concat]+individual[concatWith]))
                    individual[concatWith] = [0]
                    #print(concat)
                    #rint(concatWith)
                    #print(f'individual {individual}')
                    break
        individual = list(itertools.chain(*individual))
        individual.append(0)
        individual = np.array(individual)
        #print(individual)
        return individual

    def concatTrucksPopulation(self,population):
        mutatedPop = []
        for i in range(self.eliteSize):
            mut = population[i].copy()
            m = self.concatTrucks(mut)
            loadsm = self.initialLoads([m])
            load = self.initialLoads([population[i]])
            if self.fitnessFunc([m], loadsm)[0][1] > self.fitnessFunc([population[i]], load)[0][1]:
                mutatedPop.append(m)
            else:
                mutatedPop.append(population[i])
        for ind in range(self.eliteSize, len(population)):
            mutatedInd = self.concatTrucks(population[ind])
            mutatedPop.append(mutatedInd)
        return mutatedPop


    def nextGeneration(self,currentGen):
        loads = self.initialLoads(currentGen)
        popRanked = self.fitnessFunc(currentGen,loads)
        selectionResults = self.selection(popRanked)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.breedPopulation(matingpool)
        mutatedTrucks = self.mutateTrucksPopulation(children)
        #mutatedInverse = self.mutateInversePopulation(children)
        #print(mutatedInverse)
        mutateGens = self.mutatePopulation(mutatedTrucks)
        nextGeneration = self.concatTrucksPopulation(mutateGens)
        return nextGeneration

    def geneticAlgorithmPlot(self):
        pop = self.initial_population()
        loads = self.initialLoads(pop)
        progress = []
        progress.append(self.factor / self.fitnessFunc(pop, loads)[0][1])
        #print(self.factor / self.fitnessFunc(pop, loads)[0][1])
        #print(progress)
        #print(self.fitnessFunc(pop, loads))

        for i in tqdm(range(0, self.generations), desc=f'In Progress..'):
            pop = self.nextGeneration(pop)
            # print(pop)
            loads = self.initialLoads(pop)
            progress.append(self.factor / self.fitnessFunc(pop, loads)[0][1])
            # print(cost_func(pop))
        plt.figure(figsize=(10, 10))
        plt.plot(progress)
        plt.ylabel('Cost')
        plt.xlabel('Generation')
        bestRouteIndex = self.fitnessFunc(pop, loads)[0][0]
        bestRoute = pop[bestRouteIndex]
        self.bestSol = bestRoute
        print(f'Cost of route: {self.factor / self.fitnessFunc(pop, loads)[0][1]}')
        print(bestRoute)
        plt.show()

