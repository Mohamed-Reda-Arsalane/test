import numpy as np
from Models import *
from Utilities import *
import matplotlib.pyplot as plt

bow = splitlist(readfile("Data\\BOW.txt"))
features_bow = bow[1]
labels_bow = bow[0]
features_bow = np.array(features_bow)
print("Number of samples BOW: ", len(features_bow))
print("Intial Shape of samples BOW: ", features_bow.shape)
features_bow, labels_bow = not_null(features_bow, labels_bow)
print("Shape of samples BOW after not null: ", features_bow.shape)
features_bow_scalled = notglobalscale(features_bow)
