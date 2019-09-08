#NOTE: most of the DecisionTree class and <10 other lines were not written by me - they formed the code I extended

import numpy as np
import random
import math
import copy
import time


epsilon = 0.000000001
Epsilon = 0.00001

start_time = 0
time_limit = 300 #5 minutes

    
#algorithm to run. Options: backtrack, entropy, random, half
#TODO: add options for running approx algorithm

algorithm = "half"



class TimeLimitExceededError(Exception):
    pass



class DecisionTree(object):
    def __init__(self, n_var, n_nodes, max_depth):
        self.n_var = n_var         #NOTE Leaves are also assigned variables
        self.n_nodes = n_nodes     #NOTE This does count leaves
        self.max_depth = max_depth #NOTE This does count leaves, but assignes depth 0 to the root

        # 0 -> not available
        # 1 -> available
        self.available_nodes = set()
        self.graph = {0: [-1, -1]}
        self.available_nodes.add(0)
        self.labels = {}
        self.leafs = {}

        assert self.n_nodes % 2 == 1, "n_nodes needs to be odd"
        for i in range(0, int((self.n_nodes - 1) / 2)):
            self.add_random_node()

        self.label_graph()
        self.add_reals()

    def label_graph(self):
        available_labels = np.ones(self.n_var)
        node_stack = [0]
        variable_stack = []
        while node_stack:
            node = node_stack[-1]
            if node in self.labels:
                node_stack.pop()
                v = variable_stack.pop()
                available_labels[v] = 1
                continue
            label = np.random.choice(available_labels.nonzero()[0])
            self.labels[node] = label
            variable_stack.append(label)
            available_labels[label] = 0
            if node in self.graph and not self.isleaf(node):
                right = self.right(node)
                if right in self.graph:
                    node_stack.append(right)
                left = self.left(node)
                if left in self.graph:
                    node_stack.append(left)
    #TODO name + config at top
    def add_reals(self):
        for k, v in self.graph.items():
            if self.isleaf(k):
                #self.leafs[k] = np.random.uniform(0, 100.0)
                self.leafs[k] = np.random.randint(0, 2)

    def add_random_node(self):
        node = self.pick_free_node()
        self.available_nodes.remove(node)

        new_node_left = self.left(node)
        new_node_right = self.right(node)
        self.graph[node] = [new_node_left, new_node_right]

        self.graph[new_node_left] = [-1, -1]
        self.graph[new_node_right] = [-1, -1]
        if self.depth(new_node_left) < self.max_depth:
            self.available_nodes.add(new_node_left)
            self.available_nodes.add(new_node_right)
    
    
    
    def get_fourier(self, node_num):
        
        if self.isleaf(node_num):
            if abs(self.leafs[node_num]) < epsilon:
                return Fourier(self.n_var, {})
            else:
                return Fourier(self.n_var, {frozenset():self.leafs[node_num]})
        else:
            label = self.labels[node_num]
            fourier_left = self.get_fourier(self.left(node_num))
            fourier_right = self.get_fourier(self.right(node_num))
            series = {}
            
            for key in fourier_left.series:
                series[key] = series.get(key, 0) + (fourier_left.series[key]/2)
                series[key.union([label])] = series.get(key.union([label]), 0) + (fourier_left.series[key]/2)
                
            for key in fourier_right.series:
                series[key] = series.get(key, 0) + (fourier_right.series[key]/2)
                series[key.union([label])] = series.get(key.union([label]), 0) - (fourier_right.series[key]/2)
            
            for key in list(series.keys()):
                if abs(series[key]) < epsilon:
                    series.pop(key)                
            return Fourier(self.n_var, series)
            


    def pick_free_node(self):
        return np.random.choice(tuple(self.available_nodes))

    def left(self, node_num):
        return 2 * node_num + 1

    def right(self, node_num):
        return 2 * node_num + 2

    def depth(self, node_num):
        return int(np.log2(node_num + 1))
    
    def tree_depth(self):
        max_depth = 0
        for node in self.labels:
            max_depth = max(max_depth, self.depth(node))
        return max_depth

    def isleaf(self, node_num):
        return sum(self.graph[node_num]) == -2
    
    #evaluate the tree at key
    def __getitem__(self, key):
        curr = 0
        while not self.isleaf(curr):
            curr_var = self.labels[curr]
            curr = self.graph[curr][key[curr_var]]
        return self.leafs[curr]





class RecursiveDecisionTree:#TODO rename to node and maybe add seperate tree class holding the root
    def __init__(self, value = None, label = None, left = None, right = None):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
            
            
    def get_fourier(self, n_var):

        if self.isleaf():
            if abs(self.value) < epsilon:
                return Fourier(n_var, {})
            else:
                return Fourier(n_var, {frozenset():self.value})
        else:
            label = self.label
            fourier_left = self.left.get_fourier(n_var)
            fourier_right = self.right.get_fourier(n_var)
            series = {}
            
            for key in fourier_left.series:
                series[key] = series.get(key, 0) + (fourier_left.series[key]/2)
                series[key.union([label])] = series.get(key.union([label]), 0) + (fourier_left.series[key]/2)
                
            for key in fourier_right.series:
                series[key] = series.get(key, 0) + (fourier_right.series[key]/2)
                series[key.union([label])] = series.get(key.union([label]), 0) - (fourier_right.series[key]/2)
                
            for key in list(series.keys()):
                if abs(series[key]) < epsilon:
                    series.pop(key)     
            
            return Fourier(n_var, series)
    
    
    
    def tree_depth(self):
        if self.isleaf():
            return 0
        else:
            return (max(self.left.tree_depth(), self.right.tree_depth()) + 1)
        
        
    def node_count(self):
        if self.isleaf():
            return 1
        else:
            return (1 + self.left.node_count() + self.right.node_count())
        
    
    def isleaf(self):
        return self.value != None
    
    
    def __str__(self):
        if self.isleaf():
            return str(self.value)
        else:
            return str(self.label) + ": [" + str(self.left) + "," +  str(self.right) + "]"
        
        
    #evaluate the tree at key
    def __getitem__(self, key):
        if self.isleaf():
            return self.value
        else:
            if key[self.label] == 0:
                return self.left[key]
            elif key[self.label] == 1:
                return self.right[key]
            else:
                raise Exception("key can only contain 0s and 1s")
            
    #checks if self and other represent the same tree
    def __eq__(self, other):
        return isinstance(other, RecursiveDecisionTree) and self.value == other.value and self.label == other.label and self.left == other.left and self.right == other.right
       
       

#TODO identify sets with n-tuples
#Represents a Fourier series of a function $\mathbb{F}_2^n \rightarrow \mathbb{R}$, where $n = n_var$
#The represented Fourier series is $\sum_{F\in series.keys()} series[F]\cdot \chi_F$,
#   where $\chi_F: \mathbb{F}_2^n \rightarrow \mathbb{R}$ is given by $\chi_F(S) = (-1)^{\sum_{i=1}^n F_i\cdot S_i}$

class Fourier:
    
    def __init__(self, n_var, series):
        self.n_var = n_var
        self.series = series
        
        self.cleanup()
    
    
    def get_tree_approx(self, depth):#TODO config at the top
        
        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        self.cleanup()
        
        if len(self.series) == 0:
            return RecursiveDecisionTree(0)
        
        if len(self.series) == 1 and self.series.get(frozenset(), 0) != 0:
            return RecursiveDecisionTree(self.series.get(frozenset()))        
        
        if depth == 0:
            return RecursiveDecisionTree(self.series.get(frozenset(), 0))
    
        #label = self.split_label_approx()
        label = self.split_label()
        #label = self.split_label_entropy()
            
        (fourier_left, fourier_right) = self.split_on(label)
        
        tree = RecursiveDecisionTree()
        tree.label = label
        tree.left = fourier_left.get_tree_approx(depth - 1)
        tree.right = fourier_right.get_tree_approx(depth - 1)
        
        return tree              
            
        
    
    def get_tree_fast_backtrack_start(self):
        depth = self.degree()
        tree = None
        while(tree == None):
            tree = self.get_tree_fast_backtrack(depth)
            depth += 1
        return tree
    
    
    def get_tree_fast_backtrack(self, depth):
        # Maybe we can just greedily reduce degree and still minimize number of nodes (TODO: check)
        
        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        self.cleanup()
        
        if len(self.series) == 0:
            return RecursiveDecisionTree(0)
        
        elif len(self.series) == 1 and self.series.get(frozenset(), 0) != 0:
            return RecursiveDecisionTree(self.series.get(frozenset()))
        
        else:
            
            if depth <= 0:
                return None
            
            deg = self.degree()
            deg_reducing_variables = set(range(0, self.n_var))#can be faster if instead add all significant variables in the series to set
            for key in self.series:
                if len(key) == deg:
                    deg_reducing_variables.intersection_update(key)
            best_tree = None
            best_tree_node_count = math.inf
                    
            for var in deg_reducing_variables:
                
                (fourier_left, fourier_right) = self.split_on(var)
                assert(fourier_left.degree() <= (deg - 1) and fourier_right.degree() <= (deg - 1))
                tree_left = fourier_left.get_tree_fast_backtrack(depth - 1)
                if tree_left == None:
                    continue
                tree_right = fourier_right.get_tree_fast_backtrack(depth - 1)
                if tree_right == None:
                    continue
                
                tree = RecursiveDecisionTree(value = None, label = var, left = tree_left, right = tree_right)
                
                if tree.node_count() < best_tree_node_count:
                    best_tree = tree
                    best_tree_node_count = tree.node_count()
                    
                return tree #comment out for better node_count optimization (slow) TODO: config at the top

            if best_tree != None:
                return best_tree 
            
            if depth == self.degree():
                return None
            
            other_variables = set(range(0, self.n_var)).difference(deg_reducing_variables)
            for var in other_variables:
                (fourier_left, fourier_right) = self.split_on(var)
                assert(fourier_left.degree() == deg or fourier_right.degree() == deg)
                tree_left = fourier_left.get_tree_fast_backtrack(depth - 1)
                if tree_left == None:
                    continue
                tree_right = fourier_right.get_tree_fast_backtrack(depth - 1)
                if tree_right == None:
                    continue
                tree = RecursiveDecisionTree(value = None, label = var, left = tree_left, right = tree_right)
                return tree
            
            return None
        
                
    #Returns tree defining the same function as this Fourier series
    #Builds the tree recursively, obtaining Fourier series for children of the current node by spliting on some input argument (label)
    #The label to split on is chosen by one of the split_label algorithms
            
    def get_tree(self, depth):

        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        self.cleanup()
                
        if len(self.series) == 0:
            return RecursiveDecisionTree(0)
        
        elif len(self.series) == 1 and self.series.get(frozenset(), 0) != 0:
            return RecursiveDecisionTree(self.series.get(frozenset()))
        
        else:
            if algorithm == "half":
                label = self.split_label()
            elif algorithm == "random":
                label = self.split_label_random(depth)
            elif algorithm == "entropy":
                label = self.split_label_entropy(depth)
            else:
                raise Exception(algorithm + " is not a valid algorithm name")

            (fourier_left, fourier_right) = self.split_on(label)

            tree = RecursiveDecisionTree()
            tree.label = label
            tree.left = fourier_left.get_tree(depth)
            tree.right = fourier_right.get_tree(depth)
        
            return tree    
    
    
    def split_on(self, label):
        
        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        series_left = {}
        series_right = {}
        
        for key in self.series:
            mult = 1
            if label in key:
                mult = -1
            series_left[key.difference([label])]  =  series_left.get(key.difference([label]), 0) + self.series[key]
            series_right[key.difference([label])] = series_right.get(key.difference([label]), 0) + self.series[key] * mult
        return (Fourier(self.n_var, series_left), Fourier(self.n_var, series_right))
    
    
    def split_label_approx(self):#TODO remove(?)
        
        significant_vars = self.get_significant_vars()
        best_var = None
        lowest_penalty = math.inf
        
        for var in significant_vars:
            penalty = self.penalty(var)
            if penalty < lowest_penalty:
                best_var = var
                lowest_penalty = penalty
        
        return best_var
    
    
    def split_label(self):#TODO add most/half to name
        
        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        variable_counts = {}
        for key in self.series:
            for i in key:
                variable_counts[i] = variable_counts.get(i, 0) + 1
        label = get_max_key(variable_counts)

        return label
    
    
    def split_label_random(self, depth):
    
        significant_vars = self.get_significant_vars()
        n_candidate_labels = len(significant_vars)
        c = 0.000002
        repetitions = math.ceil(2 * n_candidate_labels * c * depth * (2 ** depth) * (depth + 2)) + 10
    
        base_mask = [-1] * self.n_var
    
        for var in significant_vars:
            
            base_mask[var] = 0            
            values_0 = self.get_possible_values(depth, base_mask.copy(), repetitions)
            values_0 = set(map(lambda x: round(x, 9),values_0))
            
            base_mask[var] = 1
            values_1 = self.get_possible_values(depth, base_mask.copy(), repetitions)
            values_1 = set(map(lambda x: round(x, 9), values_1))
            
            base_mask[var] = -1
            
            if len(values_0 & values_1) == 0:#if the intersection is empty
                return var
            
        raise Exception("Labels not pairwise distinct!")
    
    def split_label_entropy(self, depth):
        
        significant_vars = self.get_significant_vars()
                
        min_entropy = math.inf
        best_label = None
        
        for var in significant_vars:
            (fourier_left, fourier_right) = self.split_on(var)
            entropy = (fourier_left.entropy(depth) + fourier_right.entropy(depth))/2

            if entropy < min_entropy:
                best_label = var
                min_entropy = entropy
        
        return best_label
    
    
    def penalty(self, label):#TODO remove (?)
        
        (fourier_left, fourier_right) = self.split_on(label)
        return fourier_left.norm_non_constant() + fourier_right.norm_non_constant()
    
    
    def entropy(self, depth):
        
        outcome_qunatities = {}
        significant_vars = self.get_significant_vars()
        n_candidate_labels = len(significant_vars)
        c = 0.001
        number_of_trials = math.ceil(2 * n_candidate_labels * c * depth * (2 ** depth) * (depth + 2)) + 25
        
        for i in range(number_of_trials):
            
            if (time.clock() - start_time > time_limit):
                raise TimeLimitExceededError()
            
            argument = [-1] * self.n_var
            for j in range(self.n_var):
                argument[j] = random.choice([0,1])
                
            result = round(self.__getitem__(argument), 9)          
            outcome_qunatities[result] = outcome_qunatities.get(result, 0) + 1
            
        entropy = 0        
        for outcome in outcome_qunatities:
            probability = outcome_qunatities[outcome] / number_of_trials
            entropy -= probability * np.log2(probability)
            
        assert(isinstance(entropy, float))
            
        return entropy
        
    
    
    def get_possible_values(self, depth, mask, repetitions):#TODO depth necessary?
        
        obtained_values = set()
        i = 0
        for i in range(repetitions):
            
            if (time.clock() - start_time > time_limit):
                raise TimeLimitExceededError()
            
            argument = mask.copy()
            for j in range(0,self.n_var):
                if argument[j] != 0 and argument[j] != 1:
                    argument[j] = random.choice([0,1])
            value = self.__getitem__(argument)
            
            obtained_values.add(value)
        return obtained_values
    
    #returns the set of all coordinates that appear in the Fourier series
    def get_significant_vars(self):
        significant_vars = set()
        for key in self.series:
            for x in key:
                significant_vars.add(x)
        return significant_vars
    
    #Removes all coefficients that are within the error range (epsilon) from 0
    def cleanup(self):
        for key in list(self.series.keys()):
            if abs(self.series[key]) < epsilon:
                self.series.pop(key)
    
    
    '''def perturb(self):
        series2 = {}
        for key in self.series:
            if self.series[key] != 0:
                series2[key] = self.series[key] + np.random.uniform(-Epsilon, Epsilon)
        self.series = series2'''

    
    def norm(self):
        result = 0
        for key in self.series:
            result += self.series[key] * self.series[key]
        return result
    
    
    def norm_non_constant(self):
        result = 0
        for key in self.series:
            if key != set():
                result += self.series[key] * self.series[key]
        return result
    
    #returns the degree of the series (len of the largest key)
    def degree(self):
        deg = 0
        for key in self.series:
            if len(key) > deg:
                deg = len(key)
        return deg
    
    
    def __str__(self):
        return str(self.series)
        
        
        #evaluate the series on argument
    def __getitem__(self, argument):
        result = 0
        for key in self.series:
            mult = 1
            for x in key:
                if argument[x] == 1:
                    mult = -mult
            result += mult * self.series[key]
        return result
    

    def __sub__(self, other):
        series = copy.deepcopy(self.series)
        for key in other.series:
            series[key] = series.get(key, 0) - other.series[key]
        return Fourier(max(self.n_var, other.n_var), series)
            
    
    
    
 #returns the key associated with the biggest value in the dictionary 
def get_max_key(dictionary):
    maks = -1
    item = None
    for key in dictionary:
        if dictionary[key] > maks:
            maks = dictionary[key]
            item = key
    return item

def random_argument(n_var):
    return random.choices([0,1],k = n_var)


class Test:
    def __init__(self, n_var, n_nodes, depth):
        self.n_var = n_var
        self.n_nodes = n_nodes
        self.depth = depth
        self.result = None#TODO: name this average_time(?)
        self.node_error_rate = None
        self.depth_error_rate = None
        self.node_count_ratio = None #average recovered_tree_node_count/source_tree_node_count
        self.depth_ratio = None      #average recovered_tree_depth/source_tree_depth
        
    def __str__(self):
        return "[n_var = " + str(self.n_var) + ", n_nodes = " + str(self.n_nodes) + ", depth = " + str(self.depth) + "]: " + "result = " + str(self.result) + ", node_error_rate = " + str(self.node_error_rate) + ", depth_error_rate = " + str(self.depth_error_rate) + ", node_count_ratio = " + str(self.node_count_ratio) + ", depth_ratio = " + str(self.depth_ratio)





if __name__ == '__main__':#TODO entropy and random - tweak repetition numbers
    
    #list of tests to run:
    
    tests = [#Test(11,21,5),Test(50,51,9),Test(50,81,10),Test(60,101,11),
        
            #Test(11,21,5), Test(100,21,5), Test(1000,21,5), Test(10000,21,5),  
            
            #Test(11,101, 10), Test(100,101, 10), Test(1000,101, 10), Test(10000,101, 10),
            #Test(11,201, 10), Test(100,201, 10), Test(1000,201, 10), Test(10000,201, 10), 
            Test(11,401, 10), Test(100,401, 10), Test(1000,401, 10), Test(10000,401, 10), 
            Test(11,801, 10), Test(100,801, 10), Test(1000,801, 10), Test(10000,801, 10), 
            Test(11,1601, 10), Test(100,1601, 10), Test(1000,1601, 10), Test(10000,1601, 10),
            Test(11,2047, 10), Test(100,2047, 10), Test(1000,2047, 10), Test(10000,2047, 10),
            
            #Test(16,101, 15), Test(100,101, 15), Test(1000,101, 15), Test(10000,101, 15),
            #Test(16,201, 15), Test(100,201, 15), Test(1000,201, 15), Test(10000,201, 15), 
            #Test(16,401, 15), Test(100,401, 15), Test(1000,401, 15), Test(10000,401, 15), 
            #Test(16,801, 15), Test(100,801, 15), Test(1000,801, 15), Test(10000,801, 15),
            
            ##Test(21,101, 20), Test(100,101, 20), Test(1000,101, 20), Test(10000,101, 20), 
            #Test(21,201, 20), Test(100,201, 20), Test(1000,201, 20), Test(10000,201, 20), 
            #Test(21,401, 20), Test(100,401, 20), Test(1000,401, 20), Test(10000,401, 20) 
             ]

    
    for test in tests:
        seed_val = 123
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        n_trials = 100
        time_sum = 0
        node_error_count = 0
        depth_error_count = 0
        result_node_count_sum = 0
        result_depth_sum = 0

        try:    
            for i in range(n_trials):
                print(i)
            
                n_var = test.n_var       #NOTE Leaves are also assigned variables
                n_nodes = test.n_nodes   #NOTE This does count leaves
                max_depth = test.depth   #NOTE This does count leaves, but assignes depth 0 to the root
                tree = DecisionTree(n_var, n_nodes, max_depth)
                while(tree.tree_depth() != test.depth):
                    print(".", end = "")
                    tree = DecisionTree(n_var, n_nodes, max_depth)
#                print()
            
#                print(tree.graph)
#                print(tree.labels)
#                print(tree.leafs)
#                print("Depth of first tree:", tree.tree_depth())
#                print("Nodes in first tree:", n_nodes)

                fourier = tree.get_fourier(0)
                
                avg_coef={}
                for key in fourier.series:
                    avg_coef[len(key)] = avg_coef.get(len(key), 0) + fourier.series[key]
                
                #print("Fourier series:",fourier)
                #print(avg_coef)
#                print("Number of nonzero coefficients:" , len(fourier.series))
#                print("Degree:", fourier.degree())
                #print("Fourier norm:", fourier.norm())
                
                start_time = time.clock()
                
                if algorithm == "backtrack":
                    recursive_tree = fourier.get_tree_fast_backtrack_start()
                else:
                    recursive_tree = fourier.get_tree(fourier.degree())
                
                tree_recovery_time = time.clock() - start_time
                time_sum += tree_recovery_time
                result_node_count_sum += recursive_tree.node_count()
                result_depth_sum += recursive_tree.tree_depth()
                
#                print("Tree recovery time:", tree_recovery_time)
                
#                print("Degree:", fourier.degree())
                
                #print("Recursive tree:", recursive_tree)
#                print("Depth of recursive tree:", recursive_tree.tree_depth())
#                print("Nodes in recursive tree:", recursive_tree.node_count())
                
            
        #        recursive_tree_opt = fourier.get_tree_fast_backtrack_start()#TODO: not necessairly opt yet
                
                #print("Recursive tree 2:", recursive_tree_opt)
        #        print("Depth of recursive tree 2:", recursive_tree_opt.tree_depth())
        #        print("Nodes in recursive tree 2:", recursive_tree_opt.node_count())
                
        #        for i in range(20):
        #            tree_approx = fourier.get_tree_approx(i)
        #            print(i, (tree_approx.get_fourier(n_var) - fourier).norm())
            
                assert(tree.tree_depth() >= fourier.degree())
        #        assert(fourier.degree() >= recursive_tree.tree_depth())
                if (tree.tree_depth() < recursive_tree.tree_depth()):
                    depth_error_count += 1
                if (n_nodes < recursive_tree.node_count()):
                    node_error_count += 1
        #        assert(n_nodes == recursive_tree.node_count())
        #       assert(fourier.degree() == recursive_tree_opt.tree_depth())
        #        assert(n_nodes >= recursive_tree_opt.node_count())
                #assert(recursive_tree.node_count() == recursive_tree3.node_count())
        #        for i in range(10000):#TODO: config at the top
        #            argument = random_argument(n_var)
        #            assert(abs(tree[argument] - fourier[argument]) < epsilon)
        #            assert(abs(tree[argument] - recursive_tree[argument]) < epsilon)
        #            assert(abs(tree[argument] - recursive_tree_opt[argument]) < epsilon)
                    #assert(abs(tree[argument] - recursive_tree3[argument]) < 100 * Epsilon)
                    
                
            test.result = round(time_sum/n_trials, 5)
            test.node_error_rate =  round(node_error_count/n_trials, 5)
            test.depth_error_rate = round(depth_error_count/n_trials, 5)
            test.node_count_ratio = round(result_node_count_sum/(test.n_nodes * n_trials), 5)
            test.depth_ratio =      round(result_depth_sum/(test.depth * n_trials), 5)
            print(test)
        
        except TimeLimitExceededError:
            test.result = math.inf
            print(test)
            
        except MemoryError: #memory limit has to be set using "ulimit -v"
            test.result = "Memory error"
            print(test)
        
    for test in tests:
        print(test)
