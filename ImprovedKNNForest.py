import pandas as pd
import CostSensitiveID3
import random
import math
from sklearn.model_selection import KFold
import numpy as np



class KNNForest():
    def __init__(self ,train_group, test_group,K,p):
        self.train_group = train_group
        self.test_group = test_group
        self.p = p
        self.K = K
        self.id3 = CostSensitiveID3.CostSensitiveID3(self.train_group, self.test_group)

    #creating the trees as the given number "trees_number" with the given examples_list
    #for each tree which made, checks its accuarcy
    def NtreesCreator(self, examples_list, trees_number):
        tree_and_centroid_list = []
        learning_group_size = math.trunc(self.p * len(examples_list))
        iterations_counter=0
        while len(tree_and_centroid_list) != trees_number:
            sample_list = random.sample(examples_list, learning_group_size)
            classifier = self.id3.fit(sample_list, self.id3.features, self.id3.MaxIG,
                                      self.id3.findMajority(sample_list))
            accuracy = self.accuarcyOfTheClassifier(classifier)
            if accuracy > 0.97 or iterations_counter == 3:
                len(tree_and_centroid_list)
                centroid_dict = self.centroidDictCalc(sample_list)
                tree_and_centroid_list.append((classifier, centroid_dict))
                iterations_counter=0
            else:
                iterations_counter+=1

        return tree_and_centroid_list



    #calculate the centroid values of the given list
    #puts each value under its feautre name as key in a dictionary(instead of vector)
    def centroidDictCalc(self, sample_list):
        centroid_dict = {}
        for feature in self.id3.features:
            if feature != 'diagnosis':
                features_sum = sum([self.id3.train_dict[feature][example-1] for example in sample_list])
                centroid_dict[feature] = float(features_sum / len(sample_list))
        return centroid_dict


    #finds the classify of a given test list usting the given forest
    def testGroupCalc(self, forest, test_indexes, test_dict):
        correct_values_num = 0
        FP=0
        FN=0
        for test in test_indexes:
            classify = self.predictRegardingKTrees(forest,test,test_dict)
            real_classify = test_dict['diagnosis'][test - 1]
            if classify == real_classify:
                correct_values_num += 1
            else:
                if real_classify == 'B':
                    FP+=1
                else:
                    FN+=1
        accuracy = (float(float(correct_values_num) / len(test_indexes)))
        return (accuracy,FP,FN)

    # for a given test examlle and a forest,
    # returns the most comman calssify which classified by the tress
    #if the majority calssify is not a major of 75%, addes tress till reaching 75%
    def predictRegardingKTrees(self,forest,test,test_dict):
        num_B=0
        nun_M=0
        classifications_list = []
        closest_trees_and_distance = self.FindclosestTrees(forest, test, test_dict)
        k_closest_trees_and_distance = closest_trees_and_distance[:self.K]
        rest_of_trees = closest_trees_and_distance[self.K:]
        tree_list = [tree[0] for tree in k_closest_trees_and_distance]
        for tree in tree_list:
            classification = self.id3.predict(test,tree,test_dict)
            classifications_list.append(classification)
            if classification == 'B':
                num_B+=1
            else:
                nun_M+=1

        B_precentage = num_B / len(classifications_list)
        while B_precentage < 0.75 and B_precentage > 0.25 and len(rest_of_trees) > 0:
            new_tree = rest_of_trees.pop(0)[0]
            new_classification = self.id3.predict(test, new_tree, test_dict)
            classifications_list.append(new_classification)
            if new_classification == 'B':
                num_B += 1
            else:
                nun_M += 1
            B_precentage = num_B / len(classifications_list)
        if num_B > nun_M:
           return 'B'
        return 'M'


    #For a given  forest, returns a list ordered by the centroid
    # distance to the given test
    def FindclosestTrees(self, forest, test, test_dict):
        closest_trees=[]
        for tree in forest:
            distance_from_tree = self.distanceFromTree(tree[1], test,test_dict)
            closest_trees.append((tree[0],distance_from_tree))
        closest_trees.sort(key = lambda x:x[1])
        return closest_trees


    #calculate the distance from a given tree to a given test by thier centroid values
    def distanceFromTree(self, tree_centroid, test, dict):
        sum=0
        for feature in self.id3.features:
            if feature!= 'diagnosis':
                example_value = dict[feature][test-1]
                tree_value = tree_centroid[feature]
                squared_distance = (example_value-tree_value)**2
                sum+=squared_distance
        return math.sqrt(sum)



    #returns estimate accuarcy for a given calssifier
    def accuarcyOfTheClassifier(self,classifier):
        total_acuuarcy=0
        examples_list = [example for example in self.id3.examples_list if example not in classifier.examples]
        if math.trunc(0.1*len(examples_list)) == 0:
            return 1
        for i in range(5):
            sample_list = random.sample(examples_list, math.trunc(0.1*len(examples_list)))
            accuracy, FP, FN = self.id3.test_group_calc(classifier=classifier,test_indexes=sample_list,test_dict=self.id3.train_dict)
            total_acuuarcy+=accuracy
        return total_acuuarcy/5

    ###################################################helpers##############################################################
    def tuningParameters(self):
        N_optional_values = [12,13,15]
        p_optional_values = [0.5,0.55,0.6,0.65,0.7]
        accuracy_values = []
        train_group = self.id3.examples_list
        kf = KFold(n_splits=5, shuffle=True, random_state=312239395)
        max_N = 0
        max_K = 0
        max_p=0
        max_avarage_accuracy = 0
        for N in N_optional_values:
            for p in p_optional_values:
                self.p=p
                for K in range(math.trunc(0.7*N),N):
                    self.K = K
                    acuuarcy_vector = []
                    for train_index, test_index in kf.split(train_group):
                        train = [train_group[index] for index in train_index]
                        test = [train_group[index] for index in test_index]
                        forest = KNNForest.NtreesCreator(train,N)
                        accuracy, FP, FN = KNNForest.testGroupCalc(forest, test,
                                                                   KNNForest.id3.train_dict)
                        acuuarcy_vector.append(accuracy)
                    curr_accuracy_avarage = np.average(acuuarcy_vector)
                    accuracy_values.append(curr_accuracy_avarage)
                    print(N,",",K,"," ,p,"is the N,K,p and the acc is ", curr_accuracy_avarage)
                    if curr_accuracy_avarage > max_avarage_accuracy:
                        max_avarage_accuracy = curr_accuracy_avarage
                        max_N = N
                        max_K = K
                        max_p=p
        return max_N,max_K,max_p

    def tuningParameters2(self):
        P1= [1,2,3,4,5]
        p2 = [0.1,0.15,0.2,0.3]
        accuracy_values = []
        train_group = self.id3.examples_list
        kf = KFold(n_splits=5, shuffle=True, random_state=312239395)
        max_p1 = 0
        max_p2 = 0
        max_avarage_accuracy = 0
        for p1 in P1:
            self.p1 = p1
            for p2 in p2:
                    self.p2=p2
                    acuuarcy_vector = []
                    for train_index, test_index in kf.split(train_group):
                        train = [train_group[index] for index in train_index]
                        test = [train_group[index] for index in test_index]
                        forest = KNNForest.NtreesCreator(train,13)
                        accuracy, FP, FN = KNNForest.testGroupCalc(forest, test,
                                                                   KNNForest.id3.train_dict)
                        acuuarcy_vector.append(accuracy)
                    curr_accuracy_avarage = np.average(acuuarcy_vector)
                    accuracy_values.append(curr_accuracy_avarage)
                    print(p1,",",p2,"is the p1,p2 and the acc is ", curr_accuracy_avarage)
                    if curr_accuracy_avarage > max_avarage_accuracy:
                        max_avarage_accuracy = curr_accuracy_avarage
                        max_p1 = p1
                        max_p2 = p2
        return max_p1,max_p2







########################################################################################################################
if __name__ == "__main__":
    train_group = pd.read_csv('train.csv')
    test_group = pd.read_csv('test.csv')
    KNNForest = KNNForest(train_group=train_group,test_group=test_group,K=9, p=0.55)
    #trees_number,K ,p = KNNForest.tuningParameters()
    forest = KNNForest.NtreesCreator(KNNForest.id3.examples_list, 13)
    accuracy, FP, FN = KNNForest.testGroupCalc(forest, KNNForest.id3.tests_list, KNNForest.id3.test_dict)
    print(accuracy)
