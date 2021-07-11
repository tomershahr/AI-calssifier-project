import pandas as pd
import  math
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

class ID3:
    def __init__(self,train_group,test_group):
        self.train_table = train_group.to_numpy()
        self.train_dict = train_group.to_dict(orient='list')
        self.features = self.train_dict.keys()
        self.examples_list = list(range(1, len(self.train_table) + 1))

        self.tests_table = test_group.to_numpy()
        self.test_dict = test_group.to_dict(orient='list')
        self.tests_list = list(range(1,len(self.tests_table) + 1))


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
    #the classifier leraning function
    def fit(self, Examples , Features, MaxIG ,default,M=None):
        #the second condition is the Early pruning
        if len(Examples) == 0 or ( M!=None and len(Examples) < M):
            return self.Node(classify=default)
        majority_diagnos = self.findMajority(Examples)
        if self.checkIfLeaf(Examples):
            classify = majority_diagnos
            return self.Node(examples=Examples,features=Features,classify=classify)
        feature_and_threshold = self.MaxIG(Examples,Features)

        selected_feature = feature_and_threshold[0]
        threshold = feature_and_threshold[1]
        examples_smaller_than_value = [example for example in Examples if self.train_dict[selected_feature][example - 1] < threshold]
        examples_bigger_than_value = [example for example in Examples if self.train_dict[selected_feature][example - 1] >= threshold]


        first_child = self.fit(Examples=examples_smaller_than_value, Features=Features,MaxIG=MaxIG,default=majority_diagnos,M=M)
        second_child = self.fit(Examples=examples_bigger_than_value, Features=Features,MaxIG=MaxIG,default=majority_diagnos,M=M)

        childrens = (first_child,second_child)

        return self.Node(examples=Examples,features=Features,compare_value=threshold,
                         childrens=childrens,feature=selected_feature)


    #Finds the majority classify in the node
    def findMajority(self, Examples):
        healthys_list = [example for example in Examples if self.train_dict['diagnosis'][example - 1] == 'B']
        sicks_list = [example for example in Examples if self.train_dict['diagnosis'][example - 1] == 'M']
        if len(healthys_list) < len(sicks_list):
            return 'M'
        else:
            return 'B'


    def checkIfLeaf(self, examples):
        healthys_list = [example for example in examples if self.train_dict['diagnosis'][example - 1] == 'B']
        if len(healthys_list) == len(examples) or len(healthys_list) == 0:
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

        health_log_calc = math.log(health_probability,2)
        sick_log_calc = math.log(sick_probability,2)
        entropy = - health_probability*health_log_calc - sick_probability*sick_log_calc

        return entropy



    def calcThresholValues(self,Examples,f):
        example_vals_by_f = []
        for example in Examples:
            example_vals_by_f.append(self.train_dict[f][example - 1])
        example_vals_by_f.sort()
        threshold_values = []
        for i in range(len(example_vals_by_f)-1):
            threshold_values.append(float(example_vals_by_f[i]+example_vals_by_f[i+1])/2)
        return threshold_values


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


########################################testing  functions############################################
    #the given classifier clasiffies the give test
    def predict(self,test,classifier,test_dict):
        if classifier.childrens == None:
            return classifier.classification
        feature = classifier.chosen_feature
        edge_value = classifier.compare_value
        test_feature_vlaue = test_dict[feature][test-1]
        if test_feature_vlaue > edge_value:
            return self.predict(test,classifier.childrens[1],test_dict)
        else:
            return self.predict(test,classifier.childrens[0],test_dict)


    #classify group of tests
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
        accuracy = (float(float(correct_values_num) / len(test_indexes)))
        return (accuracy,FP,FN)




########################################Pruning  functions############################################

    #for running experiment fucntion - delete the # sign in the line where the function is called in main
    def experiment(self):
        M_optional_values = [4,5,10,20,40,70,120,240]
        accuracy_values=[]
        train_group = self.examples_list
        kf = KFold(n_splits=5, shuffle=True, random_state=312239395)
        max_M = 0
        max_avarage_accuracy = 0
        for M in M_optional_values:
            acuuarcy_vector=[]
            for train_index, test_index in kf.split(train_group):
                train = [train_group[index] for index in train_index]
                test = [train_group[index] for index in test_index]
                root_majority = self.findMajority(train)
                classifier = self.fit(Examples=train,Features=self.features,MaxIG=self.MaxIG,
                                      default=root_majority,M=M)
                accuracy,FP,FN = self.test_group_calc(classifier, test, self.train_dict)
                acuuarcy_vector.append(accuracy)
            curr_accuracy_avarage = np.average(acuuarcy_vector)
            accuracy_values.append(curr_accuracy_avarage)
            print(M, "is the M and the acc is ", curr_accuracy_avarage)
            if curr_accuracy_avarage > max_avarage_accuracy:
                max_avarage_accuracy = curr_accuracy_avarage
                max_M = M
        self.makeGraph(M_optional_values,accuracy_values)
        print(max_M,"is the max M")
        return self.fit(Examples=self.examples_list, Features=self.features, \
                        default=self.findMajority(self.examples_list), MaxIG=self.MaxIG, M=max_M)

    def makeGraph(self,M_optional_values,accuracy_values):
        plt.plot(M_optional_values,accuracy_values)
        plt.xlabel('M value')
        plt.ylabel('Accuracy')
        plt.title('Accuracy as function of M')
        plt.show()

########################################################################################################################
if __name__ == "__main__":
    train_group = pd.read_csv('train.csv')
    test_group = pd.read_csv('test.csv')
    id3 = ID3(train_group,test_group)
    classifier = id3.fit(id3.examples_list, id3.features, id3.MaxIG, id3.findMajority(id3.examples_list))
    accuracy, FP, FN = id3.test_group_calc(classifier,id3.tests_list,id3.test_dict)
    print(accuracy)

    #For running experiment() fucntion - delete the # sign in the line below
    #classifier = id3.experiment()



    #Lines i used for calculating the accuarcy with early pruning and loss
    #accuracy, FP, FN = id3.test_group_calc(classifier,id3.tests_list,id3.test_dict)
    #print(accuracy,"with pruning")
    #loss = (0.1 * FP + FN) / len(id3.tests_list)
    #print(loss)
