import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.svm
from sklearn.metrics import classification_report, balanced_accuracy_score,confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve,auc
import warnings
warnings.filterwarnings("ignore")

def reduce_features(solution, features):
    """""
    Reduces the dataset by selecting only the features indicated by the solution vector.
    
    Parameters:
    - solution: Binary array (1 = selected feature, 0 = ignored feature)
    - features: Full feature matrix (numpy array)

    Returns:
    - Reduced feature matrix with only selected features.
    """

    selected_elements_indices = np.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features

def classification_accuracy(labels, predictions):
    """
    Computes classification accuracy.

    Parameters:
    - labels: True class labels (numpy array)
    - predictions: Predicted class labels (numpy array)

    Returns:
    - Accuracy as a float (0-1).
    """
    correct = np.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy

def metrics(labels,predictions,classes):
    """""
    Prints classification metrics including accuracy, confusion matrix, and balanced accuracy.

    Parameters:
    - labels: True labels
    - predictions: Model predictions
    - classes: List of class names
    """

    print("Classification Report:")
    print(classification_report(labels, predictions, target_names = classes))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("Classwise Accuracy :{}".format(matrix.diagonal()/matrix.sum(axis = 1)))
    print("Balanced Accuracy Score: ",balanced_accuracy_score(labels,predictions))
    
def cal_pop_fitness(pop, train_datas, train_labels, test_datas, test_labels):
    """
    Evaluates the fitness of the population using an SVM classifier.

    Parameters:
    - pop: Population of feature selection solutions (binary matrix)
    - train_datas: Training feature set
    - train_labels: Training labels
    - test_datas: Testing feature set
    - test_labels: Testing labels

    Returns:
    - accuracies2: Fitness scores (classification accuracy)
    - predictions2: Predicted labels for the test set
    - decision2: Probability scores from the SVM
    """

    pop = np.array(pop)
    accuracies1 = np.zeros(pop.shape[0])
    accuracies2 = np.zeros(pop.shape[0])
    idx = 0

    for curr_solution in pop:
        reduced_train_features = reduce_features(curr_solution, train_datas)
        reduced_test_features = reduce_features(curr_solution, test_datas)
        X=reduced_train_features
        y=train_labels
        
        ## SVM CLASSIFIER ##
        SVM_classifier = sklearn.svm.SVC(kernel='rbf',gamma='scale',C=5000, probability=True)
        SVM_classifier.fit(X, y)
        predictions2 = SVM_classifier.predict(reduced_test_features)
        decision2 = SVM_classifier.predict_proba(reduced_test_features)

        accuracies2[idx] = classification_accuracy(test_labels, predictions2)      
        
        idx = idx + 1
    return accuracies2,predictions2,decision2

def select_mating_pool(pop, fitness, num_parents):
    """
    Selects the best individuals from the population based on fitness scores.

    Parameters:
    - pop: Current population
    - fitness: Fitness scores of each individual
    - num_parents: Number of parents to select

    Returns:
    - parents: Selected parent solutions for crossover.
    """
    parents = np.empty((num_parents, pop.shape[1]))
    print(num_parents)
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
        print(parent_num)
    return parents


def crossover(parents, offspring_size):
    """
    Performs single-point crossover between parents to generate offspring.

    Parameters:
    - parents: Selected parent solutions
    - offspring_size: Shape of offspring matrix (num_offspring, num_features)

    Returns:
    - offspring: New solutions created from crossover.
    """

    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.int32(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=2):
    """
    Applies random mutations to offspring, ensuring unique mutation sites.
    """
    offspring_crossover = offspring_crossover.astype(np.int32)
    
    for idx in range(offspring_crossover.shape[0]):
        # Get all possible mutation indices
        all_indices = np.arange(offspring_crossover.shape[1])
        
        # Randomly choose unique indices to mutate
        if num_mutations < offspring_crossover.shape[1]:
            mutation_idx = np.random.choice(all_indices, size=num_mutations, replace=False)
        else:
            # If requesting more mutations than features, mutate all features
            mutation_idx = all_indices
            
        # Flip the selected bits (0->1, 1->0)
        offspring_crossover[idx, mutation_idx] = 1 - offspring_crossover[idx, mutation_idx]

    return offspring_crossover


def check_popu(pop):
    """
    Ensures each individual in the population has at least one selected feature.

    Parameters:
    - pop: Population matrix

    Returns:
    - pop: Fixed population matrix.
    """

    for i in range(pop.shape[0]):
        p = pop[i]
        if 1 not in p:
            pop[i] = np.random.randint(low=0, high=2, size=p.shape)
    return pop
        

def plot_roc(val_label,decision_val, classes, fold, caption='ROC Curve'):
    """
    Plots and saves the ROC curve for model evaluation.

    Parameters:
    - val_label: True class labels
    - decision_val: Model probability scores
    - classes: List of class names
    - fold: Cross-validation fold index
    - caption: Plot title
    """

    num_classes=len(classes)
    plt.figure()
    
    if num_classes!=2:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            y_val = label_binarize(val_label, classes=classes)
            fpr[i], tpr[i], _ = roc_curve(y_val[:, i], decision_val[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i+1, roc_auc[i]))
    else:
        fpr,tpr,_ = roc_curve(val_label,decision_val, pos_label=2)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label='ROC curve (area=%0.2f)'%roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(caption)
    plt.legend(loc="lower right")
    plt.savefig(str(len(classes))+"Fold"+str(fold)+'.png',dpi=300)
    #plt.show()

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


#Function to calculate crowding distance
def crowding_distance(values1,values2, front):
    epsilon = 0.00001
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        kk = (max(values1)-min(values1))
        if kk==0:
          kk=epsilon
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values1[sorted1[k-1]])/kk
    for k in range(1,len(front)-1):
        kk = max(values2)-min(values2)
        if kk==0:
          kk=epsilon
        distance[k] = distance[k]+ (values2[sorted2[k+1]] - values2[sorted2[k-1]])/kk
    return distance

#First function to optimize
def function1(solution, X_train, y_train, X_test, y_test):
    """
    First function to optimize (accuracy)
    """
    from sklearn.svm import SVC
    acc_values = []
    for i in range(solution.shape[0]):
        selected_features = np.where(solution[i] == 1)[0]
        if selected_features.size == 0:
            acc_values.append(0)
            continue
        
        # Use same parameters as cal_pop_fitness
        model = SVC(kernel='rbf', gamma='scale', C=5000)
        model.fit(X_train[:, selected_features], y_train)
        acc = model.score(X_test[:, selected_features], y_test)
        acc_values.append(acc)
    return np.array(acc_values)

#Second function to optimize
def function2(solution):
    value = np.where(solution==1)[0].shape[0]
    return -value

def check_sol(sol):
    for i, s in enumerate(sol):
        if np.all(s == 0):  # If all elements are 0
            idx = np.random.randint(0, s.shape[0])
            s[idx] = 1   # Force one feature to be selected
            sol[i, :] = s
    return sol