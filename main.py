from filter_methods import *
import os
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
import argparse
import time

# Start time
overall_start_time = time.time()


# -------------------------------------------
# Argument Parsing for Command-Line Execution
# -------------------------------------------

parser = argparse.ArgumentParser()
# Path to dataset:
parser.add_argument('--path', type=str, required = True, help='Path to where the csv file of features')
# CSV header flag:
parser.add_argument('--csv_header', type=bool, default=True, help='Does csv file have headers?')
# Target column
parser.add_argument('--target_column', type=int, required=True, help='Index of the target column')
# Genetic Algorithm Hyperparameters:
parser.add_argument('--popsize', type=int, default=100, help='Population Size')
parser.add_argument('--generations', type=int, default=100, help='Number of generations')
parser.add_argument('--mutation', type=int, default=6, help='Mutation percentage')
parser.add_argument('--topk', type=int, default=None, help='topk number of features (Refer to the Paper)')
parser.add_argument('--save_fig', type=bool, default=True, help='Save the figure or not')
parser.add_argument('--num_filters', type=int, default=None, help='Number of filter methods to use (if not provided, all filters are used)')
parser.add_argument('--output_folder', type=str, default="results", help='Folder to save the results')
parser.add_argument("--stable_pop", type=str, default=None, help="Path to stable population CSV (shape=(popsize, num_features)). If provided, we skip leftover random.")
# Parse arguments

args = parser.parse_args()

# -------------------------------------------
# Load Dataset
# -------------------------------------------

csv_path = args.path

# Ensure file has `.csv` extension
if ".csv" not in csv_path:
    csv_path+=".csv"

# Load CSV data into a DataFrame to get column names
if args.csv_header is None:
    df = pd.read_csv(csv_path, header=None)
else:
    df = pd.read_csv(csv_path)

# Specify the target column index from the command-line argument
target_column = args.target_column

# Print the column name of the target column
target_column_name = df.columns[target_column]
print(f"Selected target column: {target_column_name}")

# Convert DataFrame to NumPy array for further processing
df = np.asarray(df)

# Extract feature matrix (data) and labels (target)
data = np.delete(df, target_column, axis=1)  # All columns except the target column
target = df[:, target_column]  # Target column

# Display the shapes of data and target to verify
print("Data shape:", data.shape)
print("Target shape:", target.shape)

# -------------------------------------------
# Set the topk value dynamically
# -------------------------------------------

if args.topk is None:
    topk = max(1, data.shape[1] // 2)
else:
    topk = args.topk

print(f"Using topk: {topk}")

# -------------------------------------------
# Feature Selection - Apply Filter Methods
# -------------------------------------------
# Keep an ordered list of filter functions in ascending M-score order
ordered_filter_funcs = [
    Fisher_score,          
    ANOVA_F,
    info_gain,
    chi_square,
    Dispersion_ratio,
    MAD,
    Relief,
    SCC,
    feature_selection_sim,
    ER_filter              
]

sol = []
max_filters = args.num_filters if args.num_filters is not None else len(ordered_filter_funcs)

# Pick up to 'max_filters' methods, in that order
for i in range(max_filters):
    fmethod = ordered_filter_funcs[i]
    try:
        sol.append(fmethod(data, target))
        print(f"Successfully applied filter: {fmethod.__name__}")
    except Exception as e:
        print(f"Filter {fmethod.__name__} failed: {str(e)}")

if not sol:
    warning_message = "WARNING: All filter methods failed! Using random feature selection."
    print("\033[91m" + warning_message + "\033[0m")  # Print in red
    # Create a fallback filter that just returns random ordering
    rand_order = np.random.permutation(data.shape[1])
    fallback_result = Result()  # Assuming Result is imported/defined
    fallback_result.features = data
    fallback_result.scores = np.random.random(data.shape[1])
    fallback_result.ranks = np.argsort(np.argsort(-fallback_result.scores))
    fallback_result.ranked_features = data[:, rand_order]
    sol.append(fallback_result)
# ------------------------------------------------------
# Initialize Genetic Algorithm Population
# ------------------------------------------------------

# Validate population size
if args.popsize < 10:
    pop_size = 10
    print("Population size cannot be less than 10.")
else:
    pop_size = args.popsize

max_gen = args.generations  # Maximum number of generations

# We'll call the "filter solutions" list 'sol' as in your code,
# and we use 'topk' similarly to select from them.

if args.stable_pop is not None:
    # 1) Load stable population from file => shape (pop_size, data.shape[1])
    stable_pop = np.loadtxt(args.stable_pop, delimiter=",").astype(int)
    if stable_pop.shape[0] != pop_size:
        raise ValueError(
            f"Stable pop has {stable_pop.shape[0]} rows, but pop_size={pop_size}."
        )
    if stable_pop.shape[1] != data.shape[1]:
        raise ValueError(
            f"Stable pop has {stable_pop.shape[1]} features, but dataset has {data.shape[1]}."
        )

    # 2) Build 'filter_pop' from your filter solutions: top-k features for each method
    filter_pop = np.zeros((len(sol), data.shape[1]), dtype=int)

    # Mark selected top-k features in each solution
    for i, ranking_obj in enumerate(sol):
        selected = np.where(ranking_obj.ranks <= topk)[0]
        if selected.size == 0:
            # If no feature meets the condition, select the best one
            selected = [np.argmax(ranking_obj.scores)]
        filter_pop[i, selected] = 1

    # 3) Overwrite the top 'len(sol)' rows of stable_pop with those filter solutions
    #    (if you have fewer 'sol' than pop_size, that's fine)
    stable_pop[:len(sol), :] = filter_pop[:len(sol), :]

    initial_chromosome = stable_pop.copy()
    print("Using stable_pop + filter solutions as the initial population.")

else:
    # -------------------------------------------------
    # Old logic (no stable_pop given):
    # -------------------------------------------------
    init_size = len(sol)  # same as your old code

    # Initialize chromosome population
    initial_chromosome = np.zeros((pop_size, data.shape[1]), dtype=int)

    # Mark selected top-k features in each solution
    for i in range(len(sol)):
        selected = np.where(sol[i].ranks <= topk)[0]
        if selected.size == 0:
            selected = [np.argmax(sol[i].scores)]
        initial_chromosome[i, selected] = 1

    # Fill leftover population randomly
    rand_size = pop_size - init_size
    rand_sol = np.random.randint(0, 2, size=(rand_size, data.shape[1]))
    initial_chromosome[init_size:, :] = rand_sol

# Now 'initial_chromosome' is ready for the GA loop
# -------------------------------------------
# Train-Test Split for Evaluation
# -------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=0)

# -------------------------------------------
# Evolutionary Optimization Loop
# -------------------------------------------

#pop_shape = (pop_size,num_features)
num_features = data.shape[1]
num_mutations = (int)(pop_size*num_features*args.mutation/100)
solution = initial_chromosome # Initialize solutions
solution = check_sol(solution) #Ensure at least one feature is selected by check_sol


# Default values before the loop
function1_values = function1(np.array(solution),X_train,y_train,X_test,y_test).tolist()    
function2_values = [function2(solution[i])for i in range(0,pop_size)]


gen_no=0 # Generation counter
while(gen_no<max_gen):

    # Compute fitness scores
    function1_values = function1(np.array(solution), X_train, y_train, X_test, y_test).tolist()
    function2_values = [function2(solution[i]) for i in range(pop_size)]

    # **Sort solutions by accuracy before non-dominated sorting**
    sorted_indices = np.argsort(function1_values)[::-1]  # Sort accuracy in descending order
    function1_values_sorted = [function1_values[i] for i in sorted_indices]
    function2_values_sorted = [function2_values[i] for i in sorted_indices]
    solution_sorted = [solution[i] for i in sorted_indices]

    # Perform Non-Dominated Sorting
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    print("Generation number: ",gen_no+1)
    # Compute Crowding Distance

    crowding_distance_values=[]
    for front in non_dominated_sorted_solution:
        if len(front) > 0:
         crowding_distance_values.append(crowding_distance(function1_values, function2_values, front))    
    # Generate Offspring using Crossover and Mutation
    solution2 = crossover(np.array(solution), offspring_size = (pop_size,num_features))
    solution2 = mutation(solution2, num_mutations = num_mutations)
    solution2 = check_sol(solution2)
    # Compute fitness for new offspring
    function1_values2 = function1(solution2,X_train,y_train,X_test,y_test).tolist()#[function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,pop_size)]
    # Apply Non-Dominated Sorting on Offspring
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    
    # Select new generation
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1 # Increment generation counter

# -------------------------------------------
# Plot and Save Results
# -------------------------------------------

plt.figure()
dst = args.output_folder + "/"
if not os.path.exists(dst):
    os.makedirs(dst)
    
# Extract fitness values˝
func1 = [i for i in function1_values]
func2 = [j*-1 for j in function2_values]

# Scatter plot of solutions
plt.xlabel('No.of Features Selected', fontsize=15)
plt.ylabel('Classification Accuracy', fontsize=15)
plt.scatter(func2, func1)

csv_name = csv_path.split("/")[-1]

if args.save_fig:
    plt.savefig(dst + csv_name.split('.csv')[0] + "_" + '_all solutions.png', dpi=300)
df = np.concatenate(( 
    np.expand_dims(np.asarray(func2), 1),  # feature count
    np.expand_dims(np.asarray(func1), 1)   # accuracy
), axis=1)


# Compute Pareto Front
front_f = fast_non_dominated_sort(function1_values, function2_values)

pareto_indices = front_f[0]  # list of non-dominated solution indices
rows = []
for idx in pareto_indices:
    acc = function1_values[idx]
    fcount = -function2_values[idx]  # because function2 = negative #features
    rows.append([fcount, acc])

arr = np.array(rows)
# Sort by feature count
arr = arr[np.argsort(arr[:,0])]

# Convert rows to a NumPy array if not already
arr = np.array(rows)  # shape (N,2) => [ [feature_count, accuracy], ... ]

# 1) Sort by feature_count ascending
arr = arr[np.argsort(arr[:,0])]

# 2) Group by feature_count, keep only the highest accuracy
unique_counts = np.unique(arr[:,0])
grouped = []
for c in unique_counts:
    mask = (arr[:,0] == c)
    best_acc = np.max(arr[mask, 1])  # among all points with feature_count=c
    grouped.append([c, best_acc])

# Convert to array, still sorted by feature_count
grouped = np.array(grouped)

# 3) Ensure strictly increasing accuracy
pareto_strict = []
best_so_far = -float('inf')
for fc, acc in grouped:
    if acc > best_so_far:
        pareto_strict.append([fc, acc])
        best_so_far = acc

one_per_count = np.array(pareto_strict)


plt.figure()
plt.plot(one_per_count[:,0],one_per_count[:,1], "r*-")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("No. of Features Selected")
plt.ylabel("Classification Accuracy")
if args.save_fig:
    plt.savefig(dst+csv_name.split('.csv')[0]+"_pareto.png",dpi=300)

feature_counts = [x for x in func2]
accuracies     = func1
all_solutions = np.column_stack((feature_counts, accuracies))

# Save to CSV
np.savetxt(
    dst + csv_name.split('.csv')[0] + '_all_solutions.csv',
    all_solutions,
    delimiter=",",
    header="feature_count,accuracy",
    comments="",
    fmt=["%d","%.8f"]
)

#best at each feature count
# all_solutions is shape (N, 2) -> [feature_count, accuracy]

unique_counts = np.unique(all_solutions[:, 0])
agg_rows = []

for count in unique_counts:
    # Find rows where feature_count == count
    mask = (all_solutions[:, 0] == count)
    # Among those, pick the highest accuracy
    best_acc = np.max(all_solutions[mask, 1])
    agg_rows.append([count, best_acc])

# Convert to NumPy array and sort by ascending feature count
agg_array = np.array(agg_rows)
agg_array = agg_array[np.argsort(agg_array[:, 0])]

# Save results to CSV
np.savetxt(
    dst + csv_name.split('.csv')[0] + '_per_count.csv',
    agg_array,
    delimiter=",",
    header="feature_count,accuracy",
    comments="",
    fmt=["%d","%.8f"]
)


# Save CSV
np.savetxt(
    dst + csv_name.split(".csv")[0] + "_pareto.csv",
    one_per_count,
    delimiter=",",
    header="feature_count,accuracy",
    comments="",
    fmt=['%.0f','%.8f']
)

# Save CSV
np.savetxt(
    dst + csv_name.split(".csv")[0] + "_pareto.csv",
    one_per_count,
    delimiter=",",
    header="feature_count,accuracy",
    comments="",
    fmt=['%.0f','%.8f']
)

init_chromosome_path = os.path.join(dst, csv_name.split('.csv')[0] + '_initial_chromosomes.csv')
np.savetxt(
    init_chromosome_path,
    initial_chromosome,
    delimiter=",",
    fmt="%d"
)
print(f"Initial chromosomes saved to: {init_chromosome_path}")

"""df = np.concatenate(( np.expand_dims(np.asarray(func1),1), np.expand_dims(np.asarray(func2),1) ), axis = 1)
np.savetxt(dst+csv_name.split('.csv')[0]+'_all solutions.csv',df,newline='\n', delimiter=",")

df = df[df[:,1].argsort()]
feat_unique = np.unique(df[:,1])

# Save Pareto-optimal solutions
pareto = np.array([0,1])
pareto = np.expand_dims(pareto, axis=0)
thresh = 0.0
for f in feat_unique:
    acc_li = []
    for i in range(df.shape[0]):
        if df[i,1] == f:
            acc_li.append(df[i,0])
    max_acc = max(acc_li)
    if max_acc>thresh:
        kk = np.expand_dims(np.asarray([max_acc, f]), axis=0)
        pareto = np.concatenate((pareto, kk), axis=0)
        thresh = max_acc

pareto = np.delete(pareto, 0, 0)
np.savetxt(dst+csv_name.split('.csv')[0]+"_pareto.csv", pareto, delimiter=",", newline="\n")

acc = pareto[:,0].astype(float)
fs = pareto[:,1].astype(int)

plt.figure()
plt.plot(fs,acc,"r*-")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("No. of Features Selected")
plt.ylabel("Classification Accuracy")"""


# Measure and save runtime
overall_end_time = time.time()
total_runtime = overall_end_time - overall_start_time
time_filename = os.path.join(dst, csv_name.split(".csv")[0] + "_pareto_time.txt")
with open(time_filename, "w") as f:
    f.write(f"Total runtime: {total_runtime:.2f} seconds\n")

print(f"Total runtime saved to: {time_filename}")
