import pandas as pd
from GeneticAlgorithm import GeneticAlgorithm
import numpy as np
import random
import itertools

df = pd.read_excel('C:\\Users\mbarut\PycharmProjects\Route_Parameters\Program\\2020-11-17_Parameters.xlsx',sheet_name='2020-11-17-Distance Matrix',index_col='ORI/DES')
df = df.apply(lambda x: x/1000)
data2 = pd.read_excel('C:\\Users\mbarut\PycharmProjects\Route_Parameters\Program\\2020-11-17_Parameters.xlsx',sheet_name='2020-11-17 Parametreler',index_col='DEL_DAY')
#data2 = data2.iloc[:15,:]

#print(df)
#print(data2)

DEMAND = {}
EMPTY = {}
ENCODE = {}
DELIVERY = {i:10/60 if i !=0 else 0 for i in range(len(df))}
FIX = {i:30 if i !=0 else 0 for i in range(len(df))}
#FIX = {i+1:120 if data2.iloc[i][data2.columns[7]]==120 else 30 for i in range(len(data2))}
#FIX[0] = 0
#print(FIX)
#FIX[6] = 120


for i in range(len(data2)):
  DEMAND[data2.iloc[i]['CUST_NO']] = data2.iloc[i]['EQ_DEL']
  EMPTY[data2.iloc[i]['CUST_NO']] = data2.iloc[i]['EQ_RET']
  ENCODE[i+1] = data2.iloc[i]['CUST_NO']

DEMAND['AL POLATLI'] = 0
EMPTY['AL POLATLI'] = 0
ENCODE[0] ='AL POLATLI'

#EMPTY = {i:0 for i in range(16)}

#print(ENCODE)

KmCost = [4,4,4,4]
TCAPACITY = [150,151,180,180]
USAGE_COST = [0,0,0,0]
TTYPE = ['A','A','B','B']
TALLOWED = {i+1:data2.iloc[i]['Truck Type Allowed'] for i in range(len(data2))}
TALLOWED[0] = 'A,B'

#print(GeneticAlgorithm.kmcost(np.array([0,1,0]),df,ENCODE))
ga = GeneticAlgorithm(distanceMatrix=df,popSize=100,eliteSize=20,mutationRate=0.08,tmutateRate=0.2,tmutateFlip=0.1,generations=2000,kmCost=KmCost,tCapacity=TCAPACITY,aCost=USAGE_COST,demand=DEMAND,encode=ENCODE,empty=EMPTY,ftime=FIX,vtime=DELIVERY,ttype=TTYPE,tallowed=TALLOWED)

#ga.bestSol = np.array([ 0,  13,  14,  15,  16,  17,  0,  7,  8,  9,  10,  11,  12,  0,  18,  19,  20,  21,  22,  0,  1,  2,  3,  4,  5,  6,  0])

#print(ga.fitnessFunc(ga.initial_population(),ga.initialLoads(ga.initial_population())))
#print(ga.selection(ga.initial_population()))



#print(ga.mutateInverse([ 0, 4,  3,  1,  2,  0, 10, 13,  9,  6,  7,  8,  5,  0,  0, 12, 11,  0]))

#ga.bestSol = np.array([ 0, 4,  3,  1,  2,  0, 10, 13,  9,  6,  7,  8,  5,  0,  0, 12, 11,  0])
#loads = ga.initialLoads([[ 0,  9,  6,  5,  8,  7,  0,  4,  2,  3,  1,  0,  0, 12, 13, 10, 11,  0]])
#print(loads)
#print(ga.check_capacity(1,0,loads))
#print(ga.mutateTrucksPopulation([np.array([ 0,  4,  2,  3,  1,  0, 10, 13,  6,  7,  8,  5,  9,  0,  0, 12, 11,  0])]))
print(ga.geneticAlgorithmPlot())
print(ga.check_feasibility())
print(ga.check_collectedcylinders())


#ga = GeneticAlgorithm(distanceMatrix=df,popSize=100,eliteSize=10,mutationRate=0.05,generations=50,kmCost=KmCost,tCapacity=TCAPACITY,aCost=USAGE_COST,demand=DEMAND,encode=ENCODE,empty=EMPTY)
#print(ga.geneticAlgorithmPlot())
#print(loads)
#print(GeneticAlgorithm.kmcost(np.array([0, 4, 6, 5, 3, 2, 1, 0]),distanceMatrix=df,encode=ENCODE))
#print(ga.calculate_chromosome_cost(np.array([ 0,  9,  6,  5,  8,  7,  0,  4,  2,  3,  1,  0,  0, 12, 13, 10, 11,  0]),loads,routeIndex=0))

#print(GeneticAlgorithm.kmcost(np.array([0,1,0]),df,ENCODE))

#aa = [np.array([0,1,2,3,4,5,6,0,7,8,9,10,11,12,13,0,14,15,16,17,18,19,0,0])]
#print(1/ga.fitnessFunc(aa,ga.initialLoads(aa))[0][1])

#print(ga.fitnessFunc(np.array([0,1,2,0]),loads=[[10]]))
#pop = GeneticAlgorithm(distanceMatrix=df,popSize=100,eliteSize=10,mutationRate=0.20,generations=10,kmCost=KmCost,tCapacity=TCAPACITY,aCost=USAGE_COST,demand=DEMAND,encode=ENCODE,empty=EMPTY)
#print(pop.fitnessFunc(np.array([[0,19, 13,  8, 10, 11, 12,  7, 15,0]]),[[173]]))
#print(pop.geneticAlgorithmPlot([19, 13,  8, 10, 11, 12,  7, 15]))
#print(pop.geneticAlgorithmPlot(uniqueSets=[1]))
#print(ga.geneticAlgorithmPlot())

#routes = [ 0, 0, 19, 13, 18, 17,  9,  7, 15,  0, 10,  8, 11, 12,  1,  0,  6,  5,  3,  2,  4, 14, 16,  0]




"""
def breakRoutes(individual):
  b = np.argwhere(individual == 0)
  routes = []
  for i in range(len(b) - 1):
    route = []
    for j in range(b[i][0], b[i + 1][0] + 1):
      route.append(individual[j])
    routes.append(route)
  return routes

def mutateTrucks(individual, mutationRate):
  individual = breakRoutes(individual)
  for i in individual:
    i = i.remove(0)
  individual = np.array(individual,dtype=object)
  for swapped in range(len(individual)):
    if (random.random() < mutationRate):
      swapWith = int(random.random() * len(individual))
      city1 = individual[swapped]
      city2 = individual[swapWith]
      individual[swapped] = city2
      individual[swapWith] = city1
      break
  individual = list(itertools.chain(*individual))
  individual.insert(0,0)
  individual = np.array(individual)
  return individual

#print(mutateTrucks(np.array([ 0, 0, 19, 13, 18, 17,  9,  7, 15,  0, 10,  8, 11, 12,  1,  0,  6,  5,  3,  2,  4, 14, 16,  0]),mutationRate=0.5))

"""