################################################
# decision_trees.py:
#   Operations for building and testing decision
#   trees.
# author:
#   @Froilan Luna-Lopez
#       CS 422.1001
#       University of Nevada, Reno
#       27 September 2022
################################################

# Libraries
import numpy as np
import math as m

# Global variables
featureNames = [] # Stores dummy feature names.

################################################
# binaryColSplit():
#   Split a column from a list of lists.
# args:
#   @features: List of lists with column.
#   @colNum: Column to split. First column = 0.
# return:
#   leftSplit, rightSplit
#   @leftSplit: List where column element = 0.
#   @rightSplit: List where column element = 1.
################################################
def binaryColSplit(features: list, labels: list, colNum: int):
    # Variables
    leftSplit = np.array([])
    rightSplit = np.array([])
    
    # Create left split
    for rowNum in range(len(features)):
        if features[rowNum][colNum] == 0:
            leftSplit = np.concatenate([leftSplit, [labels[rowNum]]], axis = None)
            
    # Create right split
    for rowNum in range(len(features)):
        if features[rowNum][colNum] == 1:
            rightSplit = np.concatenate([rightSplit, [labels[rowNum]]], axis = None)
            
    return leftSplit, rightSplit

################################################
# binaryDTSplit():
#   Split a list of lists based on a feature.
# args:
#   @features: List of lists to split.
#   @colNum: Feature column number to test with.
# return:
#   leftSplit, rightSplit
#   @leftSplit: Features where wanted column is
#       equal to zero.
#   @rightSplit: Features where wanted column is
#       equal to zero.
################################################
def binaryDTSplit(features: list, colNum: int):
    # Variables
    leftSplit = np.empty((0,len(features[0])), int)
    rightSplit = np.empty((0,len(features[0])), int)
    
    # Create left branch
    for row in features:
        if row[colNum] == 0:
            leftSplit = np.append(leftSplit, [row], axis = 0)
            
    # Create right branch
    for row in features:
        if row[colNum] == 1:
            rightSplit = np.append(rightSplit, [row], axis = 0)
            
    return np.delete(leftSplit, colNum, axis = 1).tolist(), np.delete(rightSplit, colNum, axis = 1).tolist()

################################################
# generateFeatNames():
#   Generate dummy feature names into global
#   variable.
# args:
#   @featureNum: Number of features to generate
#       a name for.
################################################
def generateFeatNames(featureNum: int):
    # Generate dummy column names
    global featureNames
    featureNames = []
    
    for colNum in range(featureNum):
        featureNames.append("f" + str(colNum))
            
################################################
# calcBinaryDTEntropy():
#   Calculate entropy for a binary data tree
#   feature.
# args:
#   @propTrue: Proportion of labels that are
#       true.
# return:
#   Float value for entropy.
################################################
def calcBinaryDTEntropy(propTrue: float):
    if propTrue == 1 or propTrue == 0:
        return 0
    return -(propTrue) * m.log2(propTrue) - (1 - propTrue) * m.log2(1 - propTrue)

################################################
# calcBinaryDTIG():
#   Calculate information gain for a binary data
#   tree feature.
# args:
#   @topH: Entropy for the parent.
#   @leftH: Entropy for the left child.
#   @leftProp: Proportion of features with left
#       child value.
#   @rightH: Entropy for the right child.
#   @rightProp: Proportion of features with
#       right child value.
# return:
#   Float value for information gain.
################################################
def calcBinaryDTIG(topH: float, 
                   leftH: float, leftProp: float, 
                   rightH: float, rightProp: float):
    return topH - (leftProp * leftH) - (rightProp * rightH)

################################################
# calcProp():
#   Calculates proportion while handling for
#   domain errors.
# args:
#   @num: Numerator.
#   @den: Denominator.
# return:
#   Quotient.
################################################
def calcProp(num, den):
    if not den:
        return 0
    return num / den

################################################
# getMostCommonLabel():
#   Returns the most common value in a binary
#   set of features.
# args:
#   @labels: List with values for samples.
# return:
#   Integer for most common label.
################################################
def getMostCommonLabel(labels: list):
    if sum(labels) > len(labels) / 2:
        return 1
    return 0

################################################
# testSampleSuccess():
#   Tests whether a decision tree leads a
#   sample with features to the right label.
# args:
#   @sample: Sample with feature values: list
#   @label: True sample label
#   @DT: Decision tree: list
################################################
def testSampleSuccess(sample: list, label, DT: list):
    if type(DT[0]) is str: # Test for next branch
        feat = int(DT[0][1])
        branch = int(sample[feat])
        return testSampleSuccess(sample, label, DT[branch + 1])
    elif int(DT[0]) != label: # Test for failed match
        return 0
    return 1 # Successful match

################################################
# DT_make_prediction:
#   Use a data tree to make a prediction for
#   a given sample.
# args:
#   @x: Sample with feature values: list
#   @DT: Data tree: list
# return:
#   Prediction value: int
################################################
def DT_make_prediction(x: list, DT: list):
    if type(DT[0]) is str: # Test for next branch
        feat = int(DT[0][1])
        branch = int(x[feat])
        return DT_make_prediction(x, DT[branch + 1])
    return DT[0] # Return leaf value (prediction)

################################################
# DT_test_binary():
#   Calculates the accuracy of a data tree for
#   a given set of data.
# args:
#   @X: List of lists with feature values.
#   @Y: List with values with label values.
#   @DT: Data tree as list of lists and values.
#       Can be obtained with DT_train_binary().
# return:
#   Float for accuracy.
################################################
def DT_test_binary(X: list, Y: list, DT: list):
    successes = 0
    for sampleNum in range(len(X)):
        successes += testSampleSuccess(X[sampleNum], Y[sampleNum], DT)
    return successes / len(X)

################################################
# DT_train_binary():
#   Generates a decision tree of a given max
#   depth in a 1D numpy array. Tree built using
#   information gain.
# args:
#   @X: Features in a 2D numpy array.
#   @Y: Labels in a 2D numpy array.
#   @max_depth: Max level that decision tree
#       should go to.
# return:
#   Binary tree in the form of list of lists
#   and values.
################################################
def DT_train_binary(X, Y, max_depth):
    # Stop if max_depth is zero or no more features.
    if max_depth == 0:
        return [getMostCommonLabel(Y), None, None]
    elif len(X) > 0 and len(X[0]) == 0:
        return [getMostCommonLabel(Y), None, None]
    elif len(X) == 0:
        return [getMostCommonLabel(Y), None, None]
    # Test if all labels are 0.
    elif sum(Y) == 0:
        return [0, None, None]
    # Test if all labels are 1.
    elif sum(Y) == len(Y):
        return [1, None, None]
    
    # Variables
    labelsCount = len(Y)                # Number of entries in labels.
    yesLabels = sum(Y)                  # Number of labels with 1.
    h = None                            # Entropy of labels
    bestFeat = 0                        # Column of feature with best information gain.
    bestFeatVal = 0                     # Value of best feature's information gain. 
    
    h = calcBinaryDTEntropy(yesLabels / labelsCount) # Calculate entropy
    
    # Loop through features and get max information gain.
    for colNum in range(len(X[0])): # Loop through all features/columns.
        # Variables
        yesCount = 0        # Number of samples in feature with 1.
        yesLabelsLeft = 0   # Number of samples with feature = 0 and label = 1
        yesLabelsRight = 0  # Number of samples with feature = 1 and label = 1
        
        # Count feature successes for a feature.
        for rowNum in range(len(X)): # Loop through all values in a feature.
            # Count number of 1 and 0 values.
            if X[rowNum][colNum] == 1.0:
                yesCount += 1
                if Y[rowNum] == 1: # Test if label also equals 1
                    yesLabelsRight += 1
            elif X[rowNum][colNum] == 0 and Y[rowNum] == 1:
                yesLabelsLeft += 1
                
        leftProbYes = calcProp(yesLabelsLeft, (len(X) - yesCount)) # Proportion of samples with feature = 0, label = 1
        rightProbYes = calcProp(yesLabelsRight, yesCount) # Proportion of samples with feature = 1, label = 1
        
        # Calculate branch entropies
        h_0 = calcBinaryDTEntropy(leftProbYes)  # Left branch entropy
        h_1 = calcBinaryDTEntropy(rightProbYes) # Right branch entropy
        
        IG = calcBinaryDTIG(h, h_0, 1 - (yesCount / labelsCount), h_1, yesCount / labelsCount) # Information Gain
        
        # Test if new best feature is found
        if IG > bestFeatVal:
            bestFeat = colNum
            bestFeatVal = IG
            
    # Get new feature array
    leftFeatures, rightFeatures = binaryDTSplit(X, bestFeat)
    
    # Get new label array
    leftLabels, rightLabels = binaryColSplit(X, Y, bestFeat)

    # Feature selection and bookkeeping
    if len(X[0]) != len(featureNames) or len(featureNames) == 0:
        generateFeatNames(len(X[0]))
    varName = featureNames[bestFeat]
    featureNames.pop(bestFeat)
    
    # Append feature to list and recurse to next sides.
    return [varName, 
            DT_train_binary(leftFeatures, leftLabels, max_depth - 1),
            DT_train_binary(rightFeatures, rightLabels, max_depth -1)]
    
################################################
# RF_make_prediction():
#   Given a forest of binary trees, return the
#   majority vote.
# args:
#   @X: 1D array with features: list
#   @RF: Random forest trees: list
# return:
#   Prediction.
################################################
def RF_make_prediction(x: list, RF: list):
    # Get predictions for trees in random forest
    predictions = [
        DT_make_prediction(x, i) for i in RF
    ]

    # Get majority vote in predictions
    #if sum(predictions) >= len(RF):
    #    return 1
    return predictions

################################################
# test_prediction():
#   Tests whether a prediction matches a label.
# args:
#   @y: Sample label to test: int
#   @prediction: Prediction to test: int
# return:
#   1 - Prediction is correct
#   0 - Prediction is false
################################################
def test_prediction(y: int, prediction: int):
    if y == prediction:
        return 1
    return 0

################################################
# rdmSample():
#   Randomly selects 10% of a dataset.
# args:
#   @X: 2D array with samples: list
#   @percent: Percent of samples to use: float
# return:
#   2D array with randomly selected samples.
################################################
def rdmSample(X: list, percent: float):
    arr = np.array(X) # Generate numpy array for random shuffling
    np.random.shuffle(arr) # Randomly shuffle samples
    
    # Get partition size
    if percent * len(arr) < 1:
        part = 1 # Avoid partition of size zero
    else:
        part = int(percent * len(arr)) # Percent of partition size
    
    return arr[:part] # Return partitioned, shuffled samples

################################################
# genRdmSamples():
#   Generate a list of sample sets from
#   selecting random samples from a given set.
# args:
#   @X: 2D array with sampels: list
#   @percent: Percent of samples to use: float
#   @numOfSets: Number of samples to generate
#       : int
# return:
#   List with sample sets: list
################################################
def genRdmSamples(X: list, percent: float, numOfSets: int):
    sets = []
    
    for i in range(numOfSets):
        sets.append(
            rdmSample(X, percent)
        )
        
    return sets

################################################
# RF_success_count():
#   Given a prediction and a set of random
#   forest predictions, add counter to
#   successful trees.
# args:
#   @RF: Random forest: list
#   @p: Prediction: int
# return:
#   List with random forest counter: list
################################################
def RF_success_count(RF: list, p: int):
    points = []
    for rf in RF:
        if rf == p:
            points.append(1)
        else:
            points.append(0)
    return points
            
################################################
# vecAdd():
#   Add two vectors together.
# args:
#   @vec1: First vector operand: list
#   @vec2: Second vector operand: list
# return:
#   Sum of vectors: list
################################################
def vecAdd(vec1: list, vec2: list):
    return [
        (i + j) for i, j in zip(vec1, vec2)
    ]

################################################
# RF_build_random_forest():
#   Generates binary random forest trees.
# args:
#   @X: 2D array with features: list
#   @Y: 1D array with labels: list
#   @max_depth: Maximum level to build trees to
#       : int
#   @num_of_tress: Number of trees to generate
#       : int
# return:
#   Generates binary trees: list
################################################
def RF_build_random_forest(X: list, Y: list, max_depth: int, num_of_trees: int):
    # Variables
    RF_trees = [] # List to store generated trees
    
    rdm_samples = genRdmSamples(X, .1, num_of_trees) # Stores randomly selected samples for training
    
    # Loop from 0 to num_of_trees
    for i in range(num_of_trees):
        # Variables
        fb_rowNums = np.random.randint(0,\
            len(rdm_samples[i]),\
            len(rdm_samples[i])) # Row numbers generated for bootstrapping
        fb_rows = [] # Rows generated from bootstrapping
        
        # Feature bagging using max_depth
        for j in fb_rowNums:
            fb_rows.append(rdm_samples[i][fb_rowNums[j]].tolist())
        
        # Generate and save tree
        RF_trees.append(
            DT_train_binary(fb_rows, Y, max_depth)
        )
        
    return RF_trees

################################################
# RF_test_random_forest():
#   Tests the accuracy of trees in a list.
# args:
#   @X: Features used to generate random forest
#       trees: list
#   @Y: Labels used to generate random forest
#       trees: list
#   @RF: Random forest trees: list
# return:
#   Accuracy of each tree given: list
################################################
def RF_test_random_forest(X: list, Y: list, RF: list):
    # Variables
    successes = 0
    RF_successes = [0] * (len(RF))
    
    # Test each sample on random forest
    for sampleNum in range(0, len(X)):
        RF_results = RF_make_prediction(X[sampleNum], RF) # Get results for each tree
        if sum(RF_results) > len(RF_results) / 2: # Get majority vote
            prediction = 1
        else:
            prediction = 0
            
        RF_points = RF_success_count(RF_results, prediction) # Get success results for each tree
        RF_successes = vecAdd(RF_successes, RF_points) # Add tree successes to counter
        successes += test_prediction(Y[sampleNum], prediction) # Count majority success
        
    # Convert tree_results successes to percentages
    RF_successes = [float(i / len(X)) for i in RF_successes]    
    
    for resultNum in range(len(RF_successes)):
        print("DT " + str(resultNum) + ": " + str(RF_successes[resultNum]))
    
    return successes / len(X)