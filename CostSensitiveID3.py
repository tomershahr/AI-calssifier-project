import pandas as pd
import  math
import numpy as np
from sklearn.model_selection import KFold


class CostSensitiveID3:
    def __init__(self,train_group,test_group):

        self.train_table = train_group.to_numpy()
        self.train_dict = train_group.to_dict(orient='list')
        self.features = self.train_dict.keys()
        self.examples_list = list(range(1, len(self.train_table) + 1))

        self.tests_table = test_group.to_numpy()
        self.tests_list = list(range(1, len(self.tests_table) + 1))
        self.test_dict = test_group.to_dict(orient='list')

        self.parameter=0 # A parameter i used for the parameters tuning operation



    class Node:
        def __init__(self,examples=None,features=None,
                     compare_value=None,childrens=None,feature=None,classify=None):
            self.features = features
            self.examples = examples
            self.classification = classify
            self.chosen_feature = feature
            self.compare_value = compare_value
            self.childrens = childrens




#######################################training group function#######################################
    def fit(self,Examples,Features,MaxIG,default,M=None):
        if len(Examples) == 0 or ( M!=None and len(Examples) < M):
            return self.Node(classify=default)

        classify = self.findMajority(Examples)
        if self.checkIfLeaf(Examples):
            return self.Node(examples=Examples,features=Features,classify=classify)
        feature_and_threshold = self.MaxIG(Examples,Features)

        selected_feature = feature_and_threshold[0]
        threshold = feature_and_threshold[1]
        examples_smaller_than_value = [example for example in Examples if self.train_dict[selected_feature][example - 1] < threshold]
        examples_bigger_than_value = [example for example in Examples if self.train_dict[selected_feature][example - 1] >= threshold]

        first_child = self.fit(Examples=examples_smaller_than_value, Features=Features,MaxIG=MaxIG,default=classify,M=M)
        second_child = self.fit(Examples=examples_bigger_than_value, Features=Features,MaxIG=MaxIG,default=classify,M=M)
        childrens = [first_child,second_child]

        return self.Node(examples=Examples,features=Features,compare_value=threshold,
                         childrens=childrens,feature=selected_feature,classify=classify)


    #Finds the majority classify in the node
    def findMajority(self, Examples):
        healthys_list = [example for example in Examples if self.train_dict['diagnosis'][example - 1] == 'B']
        sicks_list = [example for example in Examples if self.train_dict['diagnosis'][example - 1] == 'M']
        healthy_precentage = len(healthys_list)/len(Examples)
        if len(healthys_list) > len(sicks_list):
            return 'B'
        else:
            return 'M'


    def checkIfLeaf(self, examples):
        healthys_list = [example for example in examples if self.train_dict['diagnosis'][example - 1] == 'B']
        sick_list = [example for example in examples if self.train_dict['diagnosis'][example - 1] == 'M']
        health_probability = len(healthys_list) / len(examples)
        sick_probability = len(sick_list) / len(examples)


        # I used the line below for parameters tuning whcih led me choose 0.93
        #if health_probability>self.parameter or sick_probability>self.parameter:

        if health_probability >0.93 or sick_probability >0.93:
            return True
        return False

    #calculate the entropy of set of examples
    def calcEntropy(self,Examples):
        if len(Examples) == 0:
            return 0
        num_of_examples = len(Examples)
        healtys_list = [example for example in Examples if self.train_dict['diagnosis'][example - 1] == 'B']
        num_of_healthy = len(healtys_list)
        sick_list = [example for example in Examples if self.train_dict['diagnosis'][example - 1] == 'M']
        num_of_sick = len(sick_list)
        if num_of_sick == 0 or num_of_healthy == 0:
            return 0

        health_probability = num_of_healthy/num_of_examples
        sick_probability = num_of_sick/num_of_examples


        # I used the line below for parameters tuning whcih led me choose 0.95
        #if health_probability>self.parameter or sick_probability>self.parameter:

        if health_probability > 0.95 or sick_probability > 0.95:
            return 0


        health_log_calc = math.log(health_probability,2)
        sick_log_calc = math.log(sick_probability,2)
        entropy = - health_probability*health_log_calc - sick_probability*sick_log_calc

        return entropy



    def calcThresholValues(self,Examples,f):
        example_vals_by_f = []
        for example in Examples:
            example_vals_by_f.append(self.train_dict[f][example - 1])
        example_vals_by_f.sort()
        threshol_values = []
        for i in range(len(example_vals_by_f)-1):
            threshol_values.append(float(example_vals_by_f[i]+example_vals_by_f[i+1])/2)
        return threshol_values


    def calcIG(self, Examples, value, feature):
        examples_smaller_than_value = [example for example in Examples if self.train_dict[feature][example - 1] <= value]
        examples_bigger_than_value = [example for example in Examples if self.train_dict[feature][example - 1] > value]
        main_entropy = self.calcEntropy(Examples)

        first_child_entropy = self.calcEntropy(examples_smaller_than_value)
        second_child_entropy = self.calcEntropy(examples_bigger_than_value)


        IG = main_entropy - first_child_entropy*len(examples_smaller_than_value)/len(Examples)\
            -second_child_entropy*len(examples_bigger_than_value)/len(Examples)
        return IG


    #finds the feature and the threshold value which gives the maximal information
    def MaxIG(self,Examples,Features):
        MaxIG=0
        max_feature = None
        threshold = None
        for f in Features:
            if f != 'diagnosis':
                threshold_values = self.calcThresholValues(Examples, f)
                for threshold_value in threshold_values:
                    curr_IG = self.calcIG(Examples, threshold_value, f)
                    if curr_IG >= MaxIG:
                        MaxIG = curr_IG
                        max_feature = f
                        threshold = threshold_value
        return (max_feature,threshold)





########################################testing  functions#############################################################
    #classify a given test by a given classifier
    def predict(self,test,classifier,test_dict):
        if classifier.childrens == None:
            return classifier.classification
        feature = classifier.chosen_feature
        edge_value = classifier.compare_value
        test_feature_value = test_dict[feature][test-1]
        if test_feature_value >= edge_value:
            return self.predict(test,classifier.childrens[1],test_dict)
        else:
            return self.predict(test,classifier.childrens[0],test_dict)



    def test_group_calc(self,classifier,test_indexes,test_dict):
        correct_values_num = 0
        FP=0
        FN=0
        for test in test_indexes:
            classify = self.predict(test, classifier,test_dict)
            real_classify = test_dict['diagnosis'][test - 1]
            if classify == real_classify:
                correct_values_num += 1
            else:
                if real_classify == 'B':
                    FP+=1
                else:
                    FN+=1
        accuracy = (float(correct_values_num / len(test_indexes)))
        return (accuracy,FP,FN)


########################################helpers  functions for choising parmeters#######################################


    #Function i used for determine the Threshold Value for both "check_if_leaf" and "calcEntropy" functions
    #Each time i  chose another list below deepending the value i tuned
    #Didn't know wheather you'ld like to see it so i left it here - hope it's ok
    def tuningFindingP(self):
        parameters_checkIfLeaf = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98]
        parameters_calcEntropy = [0.93,0.94,0.95,0.96,0.97,0.98]
        train_group = self.examples_list
        kf = KFold(n_splits=5, shuffle=True, random_state=312239395)
        min_precentage = 1
        min_avarage_loss = 1
        for p in parameters_checkIfLeaf:
            print(p)
            loss_vector=[]
            self.parameter = p
            for train_index, test_index in kf.split(train_group):
                train = [train_group[index] for index in train_index]
                test = [train_group[index] for index in test_index]
                root_majority = self.findMajority(train)
                classifier = self.fit(Examples=train,Features=self.features,MaxIG=self.MaxIG,
                                      default=root_majority)
                accuracy,FP,FN = self.test_group_calc(classifier, test, self.train_dict)
                loss = (0.1 * FP + FN) / len(test)
                loss_vector.append(loss)

            curr_loss_average = np.average(loss_vector)
            if curr_loss_average < min_avarage_loss:
                min_avarage_loss = curr_loss_average
                best_paramter = p
            print("p-",p,"loss-",curr_loss_average)
        self.parameter = best_paramter
        return best_paramter
########################################################################################################################



if __name__ == "__main__":
    train_group = pd.read_csv('train.csv')
    test_group = pd.read_csv('test.csv')
    cost_sensitive_id3 = CostSensitiveID3(train_group,test_group)

    #p = cost_sensitive_id3.tuning_for_finding_p()
    #I used it for paramaters tunning

    classifier = cost_sensitive_id3.fit(cost_sensitive_id3.examples_list, cost_sensitive_id3.features, cost_sensitive_id3.MaxIG, \
                                        cost_sensitive_id3.findMajority(cost_sensitive_id3.examples_list))
    accuracy, FP, FN = cost_sensitive_id3.test_group_calc(classifier,cost_sensitive_id3.tests_list,cost_sensitive_id3.test_dict)
    loss = (0.1*FP+FN)/len(cost_sensitive_id3.tests_list)
    print(loss)

