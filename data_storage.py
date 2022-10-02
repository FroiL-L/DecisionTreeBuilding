################################################
# data_storage.py:
#   Package for converting 2D arrays of strings
#   into various types for feature and label
#   data.
#   All functions take input in the form of
#       [
#           ["variable1", "variable2", ..., "labelVar"],
#           ["num1", "num2", ..., "labelVal"],
#           ...
#       ]
# author:
#   @Froilan Luna-Lopez
#       CS 422.1001
#       University of Nevada, Reno
#       27 September 2022
################################################

# Libaries
import numpy as np

################################################
# build_nparray():
#   Converts a 2D array into a 2D numpy array of
#   data type float.
# args:
#   @data: 2D array with feature and label vectors.
# return:
#   features, labels
#   @features: Numpy 2D array with features.
#   @labels: Numpy array with labels.
################################################
def build_nparray(data):
    features = np.array([i[:-1] for i in data[1:]], dtype = np.float32) # Build numpy array of only feature values
    labels = np.array([int(i[-1]) for i in data[1:]]) # Build numpy array of only sample label values
    
    #featureNames = []
    #for colNum in range(len(data[0]) - 1):
    #    featureNames.append(data[0][colNum])
        
    return features, labels

################################################
# build_list():
#   Splits a 2D array of string values into two
#   arrays, features and labels, of type float
#   and int, respectively.
# args:
#   @data: 2D array of strings with feature and
#       label values.
# return:
#   features, labels
#   @features: 2D array of type float.
#   @labels: 1D array of type int.
################################################
def build_list(data):
    features = [[float(i) 
                 for i in list[:-1]] 
                for list in data[1:]]
    labels = [int(i[-1]) for i in data[1:]]
    
    return features, labels

################################################
# build_dict():
#   Convert a 2D array of strings to a
#   dictionary with float-type feature values
#   and int-type label values. Keys are the
#   feature variable name and values are the
#   corresponding variable values.
# args:
#   @data: 2D array of strings with feature and
#       label variables.
# return:
#   featureDict, labelDict
#   @featureDict: Dictionary with feature variable key-value pairs.
#   @labelDict: Dictionary with label variable key-value pairs.
################################################
def build_dict(data):
    featureDict = { # Dictionary with feature values
        sampleNum - 1: { # Sample #: {features dict}
            data[0][labelNum]: float(data[sampleNum][labelNum]) # feature #: feature value
            for labelNum in range(len(data[0])) # Loop through features
        } for sampleNum in range(1, len(data))  # Loop through samples
    }
    labelDict = { # Dictionary with label values
        sampleNum - 1: int(data[sampleNum][len(data[0]) - 1]) # label #: label value
            for sampleNum in range(1, len(data)) # Loop through samples
    }
    
    return featureDict, labelDict
    