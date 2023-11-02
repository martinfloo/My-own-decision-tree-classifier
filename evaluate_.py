from tree import DecisionTree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time, sys

'''
Authors:
    Brage,
    Martin
Autumn 2023
'''

#a printing function for dramatic purposes
def print_with_overwrite(text):
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

if __name__ == "__main__":
    
    df = pd.read_csv('wine_dataset.csv')
   
    #Convert datasets to numpy since we did not use pandas in our Decision tree
    #This code should work for any pandas dataframe, as long as the target values are binary,
    #and the target column is the last column

    X = df.iloc[:,:len(df.columns)-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()

    seed = 5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


    X_test, val_X, y_test, val_y = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)


    models = [
    {'impurity': 'gini', 'prune': False},
    {'impurity': 'gini', 'prune': True},
    {'impurity': 'entropy', 'prune': False},
    {'impurity': 'entropy', 'prune': True}]

    accuracy = -1

    print_with_overwrite('Training.')
    time.sleep(1)
    print_with_overwrite('Training..')
    time.sleep(1)
    print_with_overwrite('Training...')
    print('')

    for model in models:
        tree = DecisionTree(seed)
        tree.learn(X_train, y_train, impurity_measure = model['impurity'], 
        prune= model['prune'])
        performance = round(accuracy_score(val_y, tree.predict_set(val_X)), 6)
        print(f'model: {model} - validation accuracy: {performance}\n')
        
        if performance > accuracy:
            accuracy = performance
            best_model = model

    #test our best model from the validation data, we also time it
    #PS: now we time just our prediction, to time our tree building; decomment the variable under and 
    #remove the other 'our_timestart'
    #our_timestart = time.time()

    test = DecisionTree(seed)
    test.learn(X_train, y_train, impurity_measure = best_model['impurity'], 
        prune= best_model['prune'])
    our_timestart = time.time()
    our_score = round(accuracy_score(y_test, test.predict_set(X_test)), 6)

    our_timeend = time.time()
    our_time = round(our_timeend - our_timestart, 6)


    #sklearn´s decision tree classifier to compare

    sk_class = DecisionTreeClassifier(random_state = seed)
    sk_class.fit(X_train, y_train)
    
    sk_timestart = time.time()
    sk_score = round(accuracy_score(y_test, sk_class.predict(X_test)), 6)
    sk_timeend = time.time()
    sk_time = round(sk_timeend - sk_timestart, 6)

    print(f'best model: {best_model}\n')
    print(f'our classifier     - score: {our_score}\t time: {our_time}\n\nsklearn classifier - score: {sk_score}\t time: {sk_time}')