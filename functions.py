import pylab
import numpy as np
from numpy.random import choice, randint
from random import random, randint
from copy import deepcopy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import math
from time import time
from multiprocessing import Pool

#library of elements; there are three types: value, function (+,-,*,/) or graph (which will be added at the end of the file)
lib = [{"type":"value","symbol":"v"},
       {"type":"function","function":lambda a:a[0]+a[1],"symbol":"+"},
       {"type":"function","function":lambda a:a[0]-a[1],"symbol":"-"},
       {"type":"function","function":lambda a:a[0]*a[1],"symbol":"*"},
       {"type":"function","function":lambda a:a[0]/a[1],"symbol":"/"}]

class Tree:
    
    def __init__(self,graph = None):
        if graph == None:
            self.graph = []
        else:
            self.graph = graph
            
    def clone(self):
        return Tree(self.graph)
    
    #function that takes in a graph and a list of input numbers and evaluates its result
    def evaluate(self,values,index=-1):
        #this case corresponds to 
        if index == -1:
            index = self.find_output_node()
        node = self.graph[index]
        lib_entry = lib[node["lib_id"]]
        node_type = lib_entry["type"]
        if node_type == "value":
            return values[node["arg_index"]]
        else:
            args = []
            for argument in node["input"]:
                arg = self.evaluate(values,argument)
                args.append(arg)
            if node_type == "function":
                return lib_entry["function"](args)
            elif node_type == "graph":
                return eval(lib_entry["graph"](args),args)

    #returns the position of the output node within the graph
    def find_output_node(self):
        referenced = []
        for node in self.graph:
            referenced += node["input"]
        for n, node in enumerate(self.graph):
            if n not in referenced:
                return n

    #returns the number of arguments of a graph
    def number_of_arguments(self):
        n_arg = 0
        for n, node in enumerate(self.graph):
            lib_entry = lib[node["lib_id"]]
            node_type = lib_entry["type"]
            if node_type == "value":
                n_arg += 1
        return n_arg

    #returns a list with elements of the type [position of the argument in the graph,argument index in the argument vector]
    def input_info(self):
        args = []
        for n, node in enumerate(self.graph):
            lib_entry = lib[node["lib_id"]]
            node_type = lib_entry["type"]
            if node_type == "value":
                args.append([n,node["arg_index"]])
        n_args = len(args)
        arguments = [0 for i in range(n_args)]
        for arg in args:
            arguments[arg[1]] = arg[0]
        return arguments
    
    def merge_input_entries(self,i=-1,j=-1):
        n_args = self.number_of_arguments()
        if n_args <= 1:
            return Tree(self.graph)
        arg_vector = [k for k in range(0,n_args)]
        # if i is out of range pick a random value within range
        i = i if i in arg_vector else randint(0,n_args-1)
        # if j is out of range pick a random value within range which is different from i
        arg_vector.pop(i)
        j = j if j in arg_vector else choice(arg_vector)
        #index to keep
        min_index = min(sorted([i,j]))
        #index to replace with min_index
        max_index = max(sorted([i,j]))
        input_indexes = self.input_info()
        new_graph = deepcopy(self.graph)
        #remove node corresponding to max_index in the input vector
        kept_node_index = input_indexes[min_index]
        scrapped_node_index = input_indexes[max_index]
        del new_graph[scrapped_node_index]
        #update all references within graph:
        #create a mapping for the updated positions in the input vector
        input_mapping = [index if index < max_index else (min_index if index == max_index else index-1) for index in range(n_args)]
        #create a mapping for the updated positions in the graph
        if kept_node_index < scrapped_node_index:
            graph_mapping = [index if index < scrapped_node_index else (kept_node_index if index == scrapped_node_index else index-1) for index in range(len(self.graph))]
        elif kept_node_index > scrapped_node_index:
            graph_mapping = [index if index < scrapped_node_index else (kept_node_index - 1 if index == scrapped_node_index else index-1) for index in range(len(self.graph))]
        #update the references
        for node in new_graph:
            node["input"] = [ graph_mapping[n] for n in node["input"] ]
            for n in node["input"]:
                n = graph_mapping[n]
            #if it's an input node, update arg_index
            if len(node["input"]) == 0:
                node["arg_index"] = input_mapping[node["arg_index"]]
        return Tree(new_graph)
    
    #this function draws the graph
    def draw(self):
        G = nx.MultiDiGraph()
        for n, node in enumerate(self.graph):
            G.add_node(n,attr_dict=node)
        for n1, node in enumerate(self.graph):
            for i, n2 in enumerate(node["input"]):
                G.add_edge(n2, n1, attr_dict={"arg":i})
        labels = {}
        for n, node in enumerate(G.nodes(data=True)):
            symbol = lib[node[1]["lib_id"]]["symbol"]
            labels[n] = symbol if symbol != "v" else node[1]["arg_index"]
        plt.close()
        nx.draw(G,labels = labels,with_labels = True,pos=nx.spring_layout(G))
        plt.show()
    
    #plot a 1D graph of the function given some parameters
    def plot_specimen(self,values,x_range=[-10,10],variable_position=-1):
        #pylab.close()
        x = np.linspace(x_range[0],x_range[1],100)
        n_args = self.number_of_arguments()
        x_pos = variable_position if variable_position in range(n_args) else randint(0,n_args-1)
        #input_size*number_of_data_points matrix that contains a list of input vectors like [parameter1, parameter2, x_value, ...]
        value_matrix = [[(xe if v == x_pos else value) for v, value in enumerate(values)] for xe in x]
        y = np.array([ self.evaluate(value_matrix[i]) for i in range(len(value_matrix)) ])
        pylab.plot(x,y)
        pylab.show()
        
    #given a set of parameters and a graph, compute the error relative to a dataset x, y
    def error(self,parameters,variable_position,data):
        X = data["x"]
        Y = data["y"]
        e = 0
        arg = deepcopy(parameters)
        for x,y in zip(X,Y):
            arg[variable_position] = x
            e += (y-self.evaluate(arg))**2
        return e if e == e else 10**10
    
    #optimizes a given graph on a set of data points x and y
    def optimize(self,data,variable_position=-1,timeout=-1):
        start = time()
        n_args = self.number_of_arguments()
        #select a random position in the input vector for the x value
        if variable_position == -1:
            variable_position = randint(0,n_args-1)
        #seed input
        values = np.random.rand(n_args)
        #do this for a thousand steps
        EPSILON = 1
        TOLERANCE = 0.01
        iterations = 0
        MAX_ITERATIONS = 50
        while True:
            grad = gradient(lambda a: self.error(a,variable_position,data),values)
            current_error = self.error(values,variable_position,data)
            #if the accuracy goal has been reached, stop and return the input vector
            if current_error < TOLERANCE:
                break
            while True:
                iterations += 1
                #if iterations > MAX_ITERATIONS:
                if (time() - start > 10) or (iterations > MAX_ITERATIONS):
                    return {"X": variable_position, "parameters": values, "iterations": iterations, "error": current_error}
                next_error = self.error(values - EPSILON*grad,variable_position,data)
                #if the change increases the error, reduce the size of the step
                if next_error > current_error:
                    EPSILON *= 0.5
                #else just move on
                else: 
                    break
            values += -EPSILON*grad
        return {"X": variable_position, "parameters": values, "iterations": iterations, "error": current_error}
    
    def full_optimize(self,data):
        min_error = 10**10
        n_args = self.number_of_arguments()
        for n in range(n_args):
            optimization = self.optimize(data,variable_position=n)
            if optimization["error"] < min_error:
                best = optimization
        return optimization
    
    #plot the optimal fit os a graph onto a set of data points
    def plot_optimized(self,data):
        pylab.plot(data["x"],data["y"])
        error = 10**10
        for i in range(self.number_of_arguments()):
            result = self.optimize(data,i)
            if result["error"] < error:
                error = result["error"]
                values = result["parameters"]
                x_pos = result["X"]
        self.plot_specimen(values,[min(data["x"]),max(data["x"])],x_pos)
    
    #fitness of a given graph
    def unfitness(self,data):
        start = time()
        result = self.full_optimize(data)
        end = time()
        return result["error"] + 1*(end-start)

    
################################################################################################################################
#function that combines two graphs by using the output node of the second graph as one of the inputs of the first
def insert_at(donor_graph,recipient_graph,site=-1):
    recipient_args = recipient_graph.input_info()
    donor_args = donor_graph.input_info()
    if site == -1:
        site = randint(0,len(recipient_args)-1)
    site_index = recipient_args[site]
    site_arg_index = recipient_graph.graph[site_index]["arg_index"]
    donor_output = donor_graph.find_output_node()
    new_graph = [deepcopy(node) for node in recipient_graph.graph]
    new_graph.pop(site_index)
    for n, node in enumerate(new_graph):
        lib_entry = lib[node["lib_id"]]
        node_type = lib_entry["type"]
        if node_type == "function" or node_type == "graph":
            node_input = [ni for ni in node["input"]]
            for s, subnode in enumerate(node_input):
                if subnode > site_index:
                    node_input[s] = subnode - 1
                elif subnode == site_index:
                    node_input[s] = donor_output + len(new_graph)
            node["input"] = node_input
        elif node_type == "value":
            node["arg_index"] -= (1 if node["arg_index"] > site_arg_index else 0)
    processed_donor = [deepcopy(node) for node in donor_graph.graph]
    for n, node in enumerate(processed_donor):
        lib_entry = lib[node["lib_id"]]
        node_type = lib_entry["type"]
        if node_type == "function" or node_type == "graph":
            node_input = [ni for ni in node["input"]]
            node["input"] = [ni + len(new_graph) for ni in node_input]
        elif node_type == "value":
            node["arg_index"] += len(recipient_args) - 1
    return Tree(new_graph+processed_donor)

def print2Dmatrix(matrix):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in matrix]))
    
#makes updates the parameters of a graph
def gradient(function,values):
    EPSILON = 0.000000001
    value_here = np.array([ function(values) for i in range(len(values)) ])
    dx = np.array([[ EPSILON if i == j else 0 for i in range(len(values)) ] for j in range(len(values)) ])
    value_there = np.array([ function(values+dx[i]) for i in range(len(values)) ])
    grad = (value_there-value_here)/EPSILON
    return grad

#takes the items from the library and mixes them to make new graphs until it finds one that fits the data
def evolution(data):
    ACCURACY_GOAL = 0.01
    POPULATION_SIZE = 300
    MUTATION_RATE = 0.1
    BIRTH_RATE = 0.1
    CLONING_RATE = 0.05
    DEATH_RATE = MUTATION_RATE + BIRTH_RATE + CLONING_RATE
    population = []
    #create the initial population
    for i in range(POPULATION_SIZE):
        graph = choice(lib[5:])["graph"].clone()
        population.append(graph)
    #compute weights
    errors = np.array([ specimen.unfitness(data) for specimen in population ])
    #stop if minimum error is below the accuracy goal
    min_error = np.ndarray.min(errors)
    #save the best
    best_error_so_far = 10**10
    winner = np.argmin(errors)
    population[winner].draw()
    population[winner].plot_optimized(data)
    if min_error < best_error_so_far:
        best_error_so_far = min_error
        fittest_specimen = population[winner]
    if min_error < ACCURACY_GOAL:
        winner = np.argmin(errors)
        return {"specimen": population[winner], "error": errors[winner]}
    weights = errors/sum(errors)
    #weights = np.array([ 1/POPULATION_SIZE for i in range(POPULATION_SIZE) ])
    #evolve
    for generation in range(20):
        print('evolving generation {0} of 20'.format(generation))
        #compute weights
        new_population = deepcopy(population)
        temp_weights = deepcopy(weights)
        #DEATH ROUND
        #kill int(POPULATION_SIZE*DEATH_RATE) graphs
        dead = 0
        n_dead = int(POPULATION_SIZE*DEATH_RATE)
        for kill in range(n_dead):
            to_die = choice(len(new_population),p=temp_weights)
            temp_weights = np.delete(temp_weights,to_die)
            temp_weights = temp_weights/np.sum(temp_weights)
            new_population.pop(to_die)
            dead += 1
        #print("killed :{0}".format(dead))
        #CLONING ROUND
        #copy int(POPULATION_SIZE*CLONING_RATE) new graphs
        clones = 0
        n_clones = int(POPULATION_SIZE*CLONING_RATE)
        for i in range(n_clones):
            clone_index = choice(len(population))
            clone = population[clone_index].clone()
            new_population.append(clone)
            clones+=1
        #print("clones :{0}".format(clones))
        #MUTATION ROUND
        #add int(POPULATION_SIZE*MUTATION_RATE) new graphs
        mutants = 0
        n_mutants = int(POPULATION_SIZE*MUTATION_RATE)
        for i in range(n_mutants):
            mutant_index = choice(len(population))
            mutant = population[mutant_index].clone()
            mutant = mutant.merge_input_entries()
            new_population.append(mutant)
            mutants+=1
        #print("mutants :{0}".format(mutants))
        #REPRODUCTION ROUND
        #add int(POPULATION_SIZE*BIRTH_RATE) new graphs
        babies = 0
        n_babies = n_dead - (n_clones+n_mutants)
        for i in range(n_babies):
            mother_index = choice(len(population))#,p=(1-weights)/sum(1-weights))#[0]["graph"]
            mother = population[mother_index]
            father_index = choice(len(population),p=(1-weights)/sum(1-weights))#[0]["graph"]
            father = population[father_index]
            child = insert_at(father,mother)
            new_population.append(child)
            babies+=1
        #print("babies :{0}".format(babies))
        population = deepcopy(new_population)
        #recompute the errors for the newcomers
        errors = deepcopy(errors)
        first_new_index = POPULATION_SIZE-int(POPULATION_SIZE*DEATH_RATE)
        for s, specimen in enumerate(population[first_new_index:]):
            errors[s+first_new_index] = specimen.unfitness(data)
        #stop if minimum error is below the accuracy goal
        min_error = np.ndarray.min(errors)
        #save the best
        winner = np.argmin(errors)
        #population[winner].draw()
        population[winner].plot_optimized(data)
        if min_error < best_error_so_far:
            best_error_so_far = min_error
            fittest_specimen = population[winner]
        if min_error < ACCURACY_GOAL:
            winner = np.argmin(errors)
            return {"specimen": population[winner], "error": errors[winner]}
        weights = errors/sum(errors)
        #print("population size : {0}".format(len(population)))
        print("min error at generation n."+str(generation)+": "+str(np.ndarray.min(errors)))
    #recompute final errors
    errors = np.array([ specimen.unfitness(data) for specimen in population ])
    winner = np.argmin(errors)
    if errors[winner] < best_error_so_far:
            best_error_so_far = errors[winner]
            fittest_specimen = population[winner]
    return {"specimen": fittest_specimen, "error": best_error_so_far}

################################################################################################################################

#Add a graph to the lib for each operator
for n in range(1,5):
    g = {"type":"graph","graph":Tree([{"lib_id":0,"input":[],"arg_index":0},
                                 {"lib_id":0,"input":[],"arg_index":1},
                                 {"lib_id":n,"input":[0,1]}])}
    lib.append(g)