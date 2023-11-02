import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
'''
Authors:
    Brage,
    Martin
Autumn 2023
'''
class DecisionTree:
    '''
    This class implements a decision tree classifier meant for binary
    classification. The Decision tree is created based on recursive binary
    partitioning of the features from the dataset with the goal of learning
    simple decision rules inferred from the datasets features.

    Methods:
        entropy(y):

        gini(y)
    
        information_gain(feature_column, y, threshold)

        split_graph(feature_column, threshold)

        build_tree(X, y)

        learn(X, y, impurity_measure='entropy', prune=False)
            
        prune_tree(node, pruneX, pruneY)        

        predict_single(x, node=None)

        predict_set(X, node=None)
            
    '''
    def __init__(self, seed = 123):
        '''
        Initilizes decision tree instance

        Attributes:
            root_node (dict):               The root node of the decision tree
            impurity_measure (function):    The impurity measure used for node
                                             splitting.
            seed (int)                      seed for random_state parameter in train_test_split                
        '''
        self.root_node = None
        self.impurity_measure = None
        self.seed = seed
    
    def entropy(self, y):
        '''
        Calculates the entropy of the target labels

        Parameters:
            y (numpy.array):    The target labels

        Returns:
            float:  The entropy value
        alculates the entropy of the target label
        '''
        label_probabilities = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in label_probabilities if p >0])
        return entropy

    def gini(self, y):
        '''
        Calculates the gini entropy of the target labels

        Parameters:
            y (numpy.array):    The target labels

        Returns:
            float:  The gini entropy value
        '''
        label_probabilities = np.bincount(y) / len(y)
        return 1 - np.sum(label_probabilities**2)

    def information_gain(self, feature_column, y, threshold):
        '''
        Calculates the information gain for the feature and threshold given
        as input.

        Parameters:
            feature_column (numpy.array):   The feature values
            y (numpy.array):                The target labels
            threshold (float):              The Threshold for the split

        Returns:
            float: The information gain value
        '''
        #calculate the entropy of parent
        parent_impurity = self.impurity_measure(y)

        #calculate the entropy of children
        #get the left and right children
        left_child_indices, right_node_indices = self.split_graph(feature_column, threshold)
        left_child = self.impurity_measure(y[np.array(left_child_indices)])
        right_child = self.impurity_measure(y[np.array(right_node_indices)])
        child_weights = (len(left_child_indices) / len(feature_column), len(right_node_indices) / len(feature_column))
        information_gain = parent_impurity - (child_weights[0] * left_child + child_weights[1]* right_child)
        return information_gain


    def split_graph(self, feature_column, threshold):
        '''
        Splits the feature column based of the threshold value

        Parameters:
            feature_column (numpy.array):   The feature column
            threshold (float):              The threshold for the split

        Returns:
            tuple:  A tuple containing to numpy arrays. The first array gives
                    the indices of where the feature values are greater than the
                    threshold (left_node_indices). And the second array tells
                    the indices of where the feature values are less than the
                    threshold (right_node_indices).
        '''
        left_node_indices = np.argwhere(feature_column > threshold).flatten()
        right_node_indices = np.argwhere(feature_column <= threshold).flatten()
        return left_node_indices, right_node_indices

    def build_tree(self, X, y):
        '''
        Builds the three recursivly based of feature columns (X) and target
        labels (y). This is done by choosing the best information gain, split
        the graph and then set the values of that node to the current state
        and then recursivly call the function again. The recursion stops either
        when all the labels left in y is the same or if all data points have
        identical feature values.

        Parameters:
            X (numpy.array): The feature values (input data)
            y (numpy.array): The target labels

        Returns:
            dict: The decision tree node of the current state
        '''
        #if all data points have the same label, return a leaf with that label
        if (len(np.unique(y)) == 1):
            return {'leaf node': y[0]}
            
        #if all data points have identical feature values, 
        # return a leaf with the most common label 
 
        elif (X == X[0]).all():
            return {'leaf node' :np.bincount(y).argmax()}
        
        else:
            #choose feature that maximizes information gain or gini index
            best_gain = -1
            best_split_feature = None
            best_threshold = None

            for feature_index in range(X.shape[1]):
                feature_column = X[:, feature_index]
                threshold = np.mean(feature_column)
                   
                gain = self.information_gain(feature_column, y, threshold)
                
                if gain > best_gain:
                    best_split_feature = feature_index
                    best_gain = gain
                    best_threshold = threshold
         
            #column with highest gini / information gain
            beste = X[:,best_split_feature]
            
            #find indeces of rows over and under the mean of the best column
            left_index, right_index = self.split_graph(beste, beste.mean())

            #split the dataset by creating new variables and removing the indeces from above
            left_split, right_split = np.delete(X, right_index, 0), np.delete(X, left_index, 0)
            left_labels, right_labels = np.delete(y, right_index), np.delete(y, left_index)
            
            #call the entire function again with our new dataset
            most_common_label = np.bincount(np.concatenate((left_labels, right_labels))).argmax()
      
            node = {'feature index': best_split_feature,
            'threshold': best_threshold,
            'left subtree':  self.build_tree(left_split, left_labels),
            'right subtree': self.build_tree(right_split, right_labels),
            'most common label': most_common_label}
    
            return node
            
    def learn(self, X, y, impurity_measure = 'entropy', prune= False):
        '''
        Fits the decision tree to the given dataset

        Parameters:
            X (numpy.array): feature values/the input data
            y (numpy.array): the target labels
            impurity_measure (str): the impurity measure to use ('entropy','gini')
            prune (bool): Wether to prune the tree or not

        Returns:
            None (Build the decision tree classifier based of the paramters)
        '''
        if impurity_measure == 'entropy':
            self.impurity_measure = self.entropy
        else:
            self.impurity_measure = self.gini   

        X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.1, random_state=self.seed)

        self.root_node = self.build_tree(X_train, y_train)
  
        if prune is not False:
            self.prune_tree(self.root_node, X_prune, y_prune)
            

    def prune_tree(self, node, pruneX, pruneY):
        '''
        Prunes the decision tree with pruneX and pruneY data
        recursivly. The recursion goes to the bottom of the tree where the
        stopping criteria is met (if it finds the leaf node), then unwinds the
        recursion up the tree. When it "unwinds" it checks each parent node of a
        leaf node of the accuracy score if it is pruned or not. If it is, will
        the parent node turn into a leaf node with the majority class of the
        parent node.

        Parameters:
            node (dict): the current node of the decision tree
            pruneX (numpy.array): the input data for pruning
            pruneY (numpy.array): the target labels for pruning

        Returns
            None (Updates the tree with new leaf nodes if it finds prunable nodes)
        '''

        if 'leaf node' in node:
            return
            

        # Recursively call prune_tree to traverse down the tree
        self.prune_tree(node['left subtree'], pruneX, pruneY)
        self.prune_tree(node['right subtree'], pruneX, pruneY)
       
        #in recursive unwinding we can traverse from bottom to up while pruning in relation to parent accuracy
        subtree_accuracy = accuracy_score(pruneY, self.predict_set(pruneX, node))
        parent_accuracy = accuracy_score(pruneY, [node['most common label']]* len(pruneY))

        if parent_accuracy >= subtree_accuracy:   
            label = node['most common label']
            node.clear()
            node['leaf node'] = label

      
    def predict_single(self, x, node = None):
        '''
        Predicts the label for a single datapoint. It is called recursivly down
        tree until it reaches a leaf node.

        Parameters:
            X (numpy.array): The single input instance
            node (dict): The current node of the decision tree

        Returns:
            int: The predicted label
        '''
        if node is None:
            node = self.root_node
        
        if 'leaf node' in node:
            return node['leaf node']
        
        if x[node['feature index']] > node['threshold']:
            return self.predict_single(x, node['left subtree'])
        
        if x[node['feature index']] <= node['threshold']:
            return self.predict_single(x, node['right subtree'])


    def predict_set(self, X, node = None):
        '''
        Predicts a label for a set of input instances. It starts the predict
        single method to be runned recursivly.

        Parameters:
            X (numpy.array): The input
            node (dict): the current node in the decision tree

        Returns:
            numpy.array: Predicted labels for the input instances
        '''
        results = np.array([])
        if node is None:
            for i in range(len(X)):
                results = np.append(results, self.predict_single(X[i]))
        else:
            for i in range(len(X)):
                results = np.append(results, self.predict_single(X[i], node))

        return results