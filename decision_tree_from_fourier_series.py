#NOTE: Most of the DecisionTree class and <10 other lines were not written by me - they formed the code I extended
#NOTE: Some comments use LaTeX notation

#NOTE: Some settings can be chosen below, others are at the beginning of the main function

#TODO: Repetitions numbers for the random algorithm and for the computation of entropy might need to be tweaked
#TODO: Testing approximate algorithms is not fully implemented

import numpy as np
import random
import math
import copy
import time


print_debug = False

#Run tests to check algorithm correctness?
test_correctness = True


#Errors of less than epsilon will be ignored (needed due to numerical errors)
epsilon = 0.000000001

start_time = 0
time_limit = 300 #5 minutes

#Random distribution of leaf values
#The tree is generated with the values in the leafs generated independently at random from this distribution
#Options: "cts" (Uniform([0, 100])), "finite" (Uniform({0, 1}))
leaf_dist = "cts"

#Algorithm to run for the exact recovery
#Options: "backtrack", "backtrack_slow","entropy", "random", "most", "none"
#NOTE/TODO: "backtrack_slow" seems to be producing trees with optimal node count in practice. However, I do not know if that has to be the case for every input Fourier series
algorithm_exact = "most"

#Algorithm to run for the approximate recovery
#Options: "entropy", "most", "none"
algorithm_approx = "none"

class TimeLimitExceededError(Exception):
    pass



#Represents a decision tree.
#self.graph gives the structure of the tree by giving, for each node in the tree, the numbers of both its children (-1 if there is no child, 0 is the number of the root)
#self.labels gives, for each node, the number of the variable assigned to it
#self.leaves gives the real value assigned to each leaf
#The value of the function represented by a decision tree on an argument arg is obtained by following the path from the root to a leaf. At node labeled with n-th variable we look at n-th coordinate of arg.
#    If that coordinate has value 0 we go left and if it has value 1 we go right. The value of the function is the real number in the leaf we reach.
class DecisionTree(object):
    
    #Creates a random decision tree of depth at most max_depth, with n_nodes nodes, that represents a function $\mathbb{F}_2^n \rightarrow \mathbb{R}$, where $n = n_var$
    #The tree is created by consecutively adding random pairs of sibling nodes and then labeling nodes with variables and leaves with values
    def __init__(self, n_var, n_nodes, max_depth):
        self.n_var = n_var         #NOTE Leaves are also assigned variables
        self.n_nodes = n_nodes     #NOTE This does count leaves
        self.max_depth = max_depth #NOTE This does count leaves, but assigns depth 0 to the root

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
        self.add_leaf_values()
    
    
    #Randomly assigns labels(variables) to all nodes in the tree (including leaves)
    #No label will appear twice on any path from the root to any leaf
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
    
    
    #Picks values for all leaves independently at random from distribution given by leaf_dist
    def add_leaf_values(self):
        for k, v in self.graph.items():
            if self.isleaf(k):
                if leaf_dist == "cts":
                    self.leafs[k] = np.random.uniform(0, 100.0)
                elif leaf_dist == "finite":
                    self.leafs[k] = np.random.randint(0, 2)
                else:
                    raise Exception(leaf_dist + " is not a valid distribution name")

    
    #Picks random node with no children and adds its 2 children to the tree
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
    
    
    #Returns a Fourier series (instance of the Fourier class) that defines the same function as the subdecision tree rooted at the node with number node_num (argument)
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
            
            fourier = Fourier(self.n_var, series)
            fourier.cleanup()
                    
            return fourier
            

    #Returns the number of a random leaf
    def pick_free_node(self):
        return np.random.choice(tuple(self.available_nodes))
    
    
    #Returns the number of the left child of the node with number node_num (or the number that child would have if it existed)
    def left(self, node_num):
        return 2 * node_num + 1
    
    
    #Returns the number of the right child of the node with number node_num (or the number that child would have if it existed)
    def right(self, node_num):
        return 2 * node_num + 2
    
    
    #Returns depth of the node with number node_num
    def depth(self, node_num):
        return int(np.log2(node_num + 1))
    
    
    def tree_depth(self):
        max_depth = 0
        for node in self.labels:
            max_depth = max(max_depth, self.depth(node))
        return max_depth


    def isleaf(self, node_num):
        return sum(self.graph[node_num]) == -2
    
    
    #Evaluate the tree at argument
    def __getitem__(self, argument):
        curr = 0
        while not self.isleaf(curr):
            curr_var = self.labels[curr]
            curr = self.graph[curr][argument[curr_var]]
        return self.leafs[curr]



#Represents a decision tree recursively.
#self.value is the real number in the root if the root is a leaf and None otherwise
#self.label is the number of the variable assigned to the root (can be None if the root is a leaf)
#self.left is the left subtree (can be None)
#self.right is the right subtree (can be None)
#The value of the function represented by a decision tree on an argument arg is obtained by following the path from the root to a leaf. At node labeled with n-th variable we look at n-th coordinate of arg.
#    If that coordinate has value 0 we go left and if it has value 1 we go right. The value of the function is the real number in the leaf we reach.
class RecursiveDecisionTree:
    
    def __init__(self, value = None, label = None, left = None, right = None):
        self.value = value
        self.label = label
        self.left = left
        self.right = right
            
            
    #Returns a Fourier series (instance of the Fourier class) that defines the same function as the decision tree defined by this object
    #n_var should be >= the highest number of variable in the tree
    def get_fourier(self, n_var):

        if self.isleaf():
            if abs(self.value) < epsilon:
                return Fourier(n_var, {})
            else:
                return Fourier(n_var, {frozenset():self.value})
        else:
            label = self.label
            assert label <= n_var, "n_var is too small - the tree uses variables with higher numbers"
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
        
        
    #Evaluate the tree at argument
    def __getitem__(self, argument):
        if self.isleaf():
            return self.value
        else:
            if argument[self.label] == 0:
                return self.left[argument]
            elif argument[self.label] == 1:
                return self.right[argument]
            else:
                raise Exception("argument can only contain 0s and 1s")
            
    #Checks if self and other represent the same tree
    def __eq__(self, other):
        return isinstance(other, RecursiveDecisionTree) and self.value == other.value and self.label == other.label and self.left == other.left and self.right == other.right
       
       

#Represents a Fourier series of a function $\mathbb{F}_2^n \rightarrow \mathbb{R}$, where $n = n_var$ (note that below we identify elements of \mathbb{F}_2^n with subsets of {1, 2, ..., n})
#The represented Fourier series is $\sum_{F\in series.keys()} series[F]\cdot \chi_F$, 
#    where $\chi_F: \mathbb{F}_2^n \rightarrow \mathbb{R}$ is given by $\chi_F(S) = (-1)^{\sum_{i=1}^n F_i\cdot S_i}$
class Fourier:
    
    def __init__(self, n_var, series):
        self.n_var = n_var
        self.series = series
        
        self.cleanup()
    
    
    #Returns a tree of depth at most depth (argument) that approximates the function given by this Fourier series
    #Builds the tree from top to bottom, obtaining Fourier series for children of the current node by conditioning on some coordinate of the input (label for the created node)
    #The label to split on is chosen by one of the split_label algorithms 
    def get_tree_approx(self, depth):
        
        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        self.cleanup()
        
        if len(self.series) == 0:
            return RecursiveDecisionTree(0)
        
        if len(self.series) == 1 and self.series.get(frozenset(), 0) != 0:
            return RecursiveDecisionTree(self.series.get(frozenset()))        
        
        if depth == 0:
            return RecursiveDecisionTree(self.series.get(frozenset(), 0))
        
        if algorithm_approx == "most":
            label = self.split_label_most()
        elif algorithm_approx == "entropy":
            label = self.split_label_entropy()
        else: 
            raise Exception(algorithm_approx + " is not a valid approximate algorithm name")
        
        (fourier_left, fourier_right) = self.split_on(label)
        
        tree = RecursiveDecisionTree()
        tree.label = label
        tree.left = fourier_left.get_tree_approx(depth - 1)
        tree.right = fourier_right.get_tree_approx(depth - 1)
        
        return tree              
            
        
    #Runs the backtrack algorithm for increasing goal depths, starting at self.degree(), until a tree is found
    def get_tree_fast_backtrack_start(self):
        depth = self.degree()
        tree = None
        while(tree == None):
            tree = self.get_tree_fast_backtrack(depth)
            depth += 1
        return tree
    
    
    #Returns a tree defining the same function as this Fourier series if one of depth not greater that depth (argument) exist, otherwise returns None 
    #Searches for a tree by trying various coordinates/variables/labels to condition on and cleverly backtracking
    #   Drops branches of the backtrack where the tree cannot be found (if the degree of the series is greater than the goal depth)
    #This algorithm works similarly to A*
    def get_tree_fast_backtrack(self, depth):
        
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
            deg_reducing_variables = self.get_significant_vars()
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
                    
                if algorithm_exact == "backtrack": #If algorithm_exact == "backtrack_slow" we try to search for a tree with few nodes instead of any tree
                    return tree

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
        
                
    #Returns a tree defining the same function as this Fourier series
    #Builds the tree from top to bottom, obtaining Fourier series for children of the current node by conditioning on some coordinate of the input (label for the created node)
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
            if algorithm_exact == "most":
                label = self.split_label_most()
            elif algorithm_exact == "random":
                label = self.split_label_random(depth)
            elif algorithm_exact == "entropy":
                label = self.split_label_entropy(depth)
            else:
                raise Exception(algorithm_exact + " is not a valid exact algorithm name")

            (fourier_left, fourier_right) = self.split_on(label)

            tree = RecursiveDecisionTree()
            tree.label = label
            tree.left = fourier_left.get_tree(depth)
            tree.right = fourier_right.get_tree(depth)
        
            return tree    
    
    #Conditions this Fourier series on the coordinate corresponding to label (argument)
    #Returns the Fourier series obtained by conditioning that coordinate to be 0 (left series) and 1 (right series)
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
    
    
    #Returns a label that appears in the highest number of keys corresponding to nonzero coefficients of the series
    def split_label_most(self):
        
        if (time.clock() - start_time > time_limit):
            raise TimeLimitExceededError()
        
        variable_counts = {}
        for key in self.series:
            for i in key:
                variable_counts[i] = variable_counts.get(i, 0) + 1
        label = get_max_key(variable_counts)

        return label
    
    
    #Returns a label such that conditioning on it gives two functions that do not share any values (with high probability)
    def split_label_random(self, depth):
    
        significant_vars = self.get_significant_vars()
        n_candidate_labels = len(significant_vars)
        c = 0.000002
        repetitions = math.ceil(2 * n_candidate_labels * c * depth * (2 ** depth) * (depth + 2)) + 10
    
        base_mask = [-1] * self.n_var
    
        for var in significant_vars:
            
            base_mask[var] = 0            
            values_0 = self.get_possible_values(base_mask.copy(), repetitions)
            values_0 = set(map(lambda x: round(x, 9),values_0))
            
            base_mask[var] = 1
            values_1 = self.get_possible_values(base_mask.copy(), repetitions)
            values_1 = set(map(lambda x: round(x, 9), values_1))
            
            base_mask[var] = -1
            
            if len(values_0 & values_1) == 0:#if the intersection is empty
                return var
            
        raise Exception("Labels not pairwise distinct!")
    
    
    #Returns a label such that conditioning on it maximizes the (estimated) information gain
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
    
    
    #Estimates the entropy of the function defined by this Fourier series (under uniform distribution on inputs)
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
        
    
    #Returns a set of all values that we obtain from this series in repetitions (argument) random trials by randomly setting each -1 in mask (argument) to 0 or 1 without changing 0's and 1's already there
    def get_possible_values(self, mask, repetitions):
        
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
    
    
    #Returns the set of all coordinates whose value can affect the value of the Fourier series
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


    #Returns the sum of the squares of the coefficients of this Fourier series
    def norm(self):
        result = 0
        for key in self.series:
            result += self.series[key] * self.series[key]
        return result
    
    
    #Returns the sum of the squares of the nonconstant coefficients of this Fourier series
    def norm_nonconstant(self):
        result = 0
        for key in self.series:
            if key != set():
                result += self.series[key] * self.series[key]
        return result
    
    
    #Returns the degree of this Fourier series (len of the largest key)
    def degree(self):
        deg = 0
        for key in self.series:
            if len(key) > deg:
                deg = len(key)
        return deg
    
    
    def __str__(self):
        return str(self.series)
        
        
        #Evaluate the series on argument
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
            
    
    
  
#Returns the key associated with the biggest value in the dictionary 
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


#Class used for running tests
#To create a test add a new member of this class to the list of tests and choose the number of variables, number of nodes and depth of the tree
#After a test is run in the main loop the variables representing the result will be set to the obtained values
class Test:
    def __init__(self, n_var, n_nodes, depth):
        self.n_var = n_var
        self.n_nodes = n_nodes
        self.depth = depth
        self.average_time = None
        self.node_error_rate = None
        self.depth_error_rate = None
        self.node_count_ratio = None #average recovered_tree_node_count/source_tree_node_count
        self.depth_ratio = None      #average recovered_tree_depth/source_tree_depth
        
    def __str__(self):
        return "[n_var = " + str(self.n_var) + ", n_nodes = " + str(self.n_nodes) + ", depth = " + str(self.depth) + "]: " + "average_time = " + str(self.average_time) + ", node_error_rate = " + str(self.node_error_rate) + ", depth_error_rate = " + str(self.depth_error_rate) + ", node_count_ratio = " + str(self.node_count_ratio) + ", depth_ratio = " + str(self.depth_ratio)





if __name__ == '__main__':
    
    #list of tests to run:
    
    tests = [#Test(11,21,5),Test(50,51,9),Test(50,81,10),Test(60,101,11),
        
            #Test(11,21,5), Test(100,21,5), Test(1000,21,5), Test(10000,21,5),  
            
            Test(11,101, 10), Test(100,101, 10), Test(1000,101, 10), Test(10000,101, 10),
            Test(11,201, 10), Test(100,201, 10), Test(1000,201, 10), Test(10000,201, 10), 
            Test(11,401, 10), Test(100,401, 10), Test(1000,401, 10), Test(10000,401, 10), 
            Test(11,801, 10), Test(100,801, 10), Test(1000,801, 10), Test(10000,801, 10), 
            Test(11,1601, 10), Test(100,1601, 10), Test(1000,1601, 10), Test(10000,1601, 10),
            Test(11,2047, 10), Test(100,2047, 10), Test(1000,2047, 10), Test(10000,2047, 10),
            
            #Test(16,101, 15), Test(100,101, 15), Test(1000,101, 15), Test(10000,101, 15),
            #Test(16,201, 15), Test(100,201, 15), Test(1000,201, 15), Test(10000,201, 15), 
            #Test(16,401, 15), Test(100,401, 15), Test(1000,401, 15), Test(10000,401, 15), 
            #Test(16,801, 15), Test(100,801, 15), Test(1000,801, 15), Test(10000,801, 15),
            
            #Test(21,101, 20), Test(100,101, 20), Test(1000,101, 20), Test(10000,101, 20), 
            #Test(21,201, 20), Test(100,201, 20), Test(1000,201, 20), Test(10000,201, 20), 
            #Test(21,401, 20), Test(100,401, 20), Test(1000,401, 20), Test(10000,401, 20) 
             ]

    
    for test in tests:
        
        n_trials = 100        
        seed_val = 123
        np.random.seed(seed_val)
        random.seed(seed_val)        
        
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
                max_depth = test.depth   #NOTE This does count leaves, but assigns depth 0 to the root
                tree = DecisionTree(n_var, n_nodes, max_depth)
                while(tree.tree_depth() != test.depth):
                    print(".", end = "")
                    tree = DecisionTree(n_var, n_nodes, max_depth)
                
                if print_debug:
                    print(tree.graph)
                    print(tree.labels)
                    print(tree.leafs)
                    print("Depth of first tree:", tree.tree_depth())
                    print("Nodes in first tree:", n_nodes)

                fourier = tree.get_fourier(0)
                fourier_copy = copy.deepcopy(fourier)
                
                if print_debug:
                    print("Fourier series:",fourier)
                    print("Number of nonzero coefficients:" , len(fourier.series))
                    print("Degree:", fourier.degree())
                    print("Fourier norm:", fourier.norm())
                
                if test_correctness:                    
                    assert(tree.tree_depth() >= fourier.degree())                
                    for i in range(10000):
                        argument = random_argument(n_var)
                        assert(abs(tree[argument] - fourier[argument]) < epsilon)                
                
                if algorithm_exact != "none":
                    
                    recursive_tree = None
                    
                    start_time = time.clock()
                
                    if algorithm_exact == "backtrack" or algorithm_exact == "backtrack_slow":
                        recursive_tree = fourier.get_tree_fast_backtrack_start()
                    else:
                        #TODO: The argument here (fourier.degree()) should be the depth of the tree, but we don't know that (though we could try values until a right one is found). However, this should usually be close enough
                        recursive_tree = fourier.get_tree(fourier.degree())
                
                    tree_recovery_time = time.clock() - start_time
                
                    time_sum += tree_recovery_time
                    result_node_count_sum += recursive_tree.node_count()
                    result_depth_sum += recursive_tree.tree_depth()
                    if (tree.tree_depth() < recursive_tree.tree_depth()):
                        depth_error_count += 1
                    if (n_nodes < recursive_tree.node_count()):
                        node_error_count += 1
        
                    if print_debug:    
                        print("Recursive tree:", recursive_tree)
                        print("Depth of recursive tree:", recursive_tree.tree_depth())
                        print("Nodes in recursive tree:", recursive_tree.node_count())
                
                    if test_correctness:
                        for i in range(10000):
                            argument = random_argument(n_var)
                            assert(abs(tree[argument] - recursive_tree[argument]) < epsilon)

                if algorithm_approx != "none":
                    for i in range(20):
                        tree_approx = fourier.get_tree_approx(i)
                        print(i, (tree_approx.get_fourier(n_var) - fourier).norm())
                  
                if test_correctness:                    
                    assert(fourier.n_var == fourier_copy.n_var and fourier.series == fourier_copy.series)
                
            if algorithm_exact != "none":    
                test.average_time = round(time_sum/n_trials, 5)
                test.node_error_rate =  round(node_error_count/n_trials, 5)
                test.depth_error_rate = round(depth_error_count/n_trials, 5)
                test.node_count_ratio = round(result_node_count_sum/(test.n_nodes * n_trials), 5)
                test.depth_ratio =      round(result_depth_sum/(test.depth * n_trials), 5)
            print(test)
        
        except TimeLimitExceededError:
            test.average_time = "Time limit exceeded!"
            print(test)
            
        except MemoryError: #NOTE: Memory limit has to be set using "ulimit -v"
            test.average_time = "Memory limit exceeded!"
            print(test)
        
    for test in tests:
        print(test)
