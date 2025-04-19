import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
import pandas as pd
from scipy.stats import spearmanr
from ReliefF import ReliefF
from joblib import Parallel, delayed
import logging
import time
import cProfile
import pstats
from functools import wraps, lru_cache
from datetime import datetime
import os
from numba import jit
from sklearn.feature_selection import f_classif

# -------------------------------------------
# Logging Setup
# -------------------------------------------

def setup_logging():
    """
    Sets up logging configuration for tracking feature selection processes.
    
    Creates a `logs` directory if it does not exist, and logs messages
    to both a file and the console.
    
    Returns:
    - logger: Configured logging object.
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/feature_selection_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger('feature_selection')

# Initialize logger
logger = setup_logging()

# -------------------------------------------
# Decorators for Profiling & Timing
# -------------------------------------------

def time_function(func):
    """
    Decorator to measure execution time of a function.
    Logs the duration after the function completes.

    Parameters:
    - func: Function to be timed.

    Returns:
    - Wrapped function with timing enabled.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Finished {func.__name__} in {duration:.2f} seconds")
        return result
    return wrapper

def profile_function(output_file=None):
    """
    Decorator to profile a function’s performance.

    Parameters:
    - output_file: If provided, saves profiling data to a file.

    Returns:
    - Wrapped function with profiling enabled.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            try:
                return profiler.runcall(func, *args, **kwargs)
            finally:
                if output_file is not None:
                    if not os.path.exists('profiles'):
                        os.makedirs('profiles')
                    stats = pstats.Stats(profiler)
                    stats.sort_stats('cumulative')
                    stats.dump_stats(f'profiles/{func.__name__}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.prof')
        return wrapper
    return decorator

#-------------------------  Utilities  -------------------------------------------#
def normalize(vector, lb=0, ub=1):
    """
    Normalizes a vector to a specified range [lb, ub].

    Parameters:
    - vector: Numpy array to be normalized.
    - lb: Lower bound (default: 0).
    - ub: Upper bound (default: 1).

    Returns:
    - Normalized numpy array.
    """
    minimum = np.min(vector)
    maximum = np.max(vector)

    if maximum == minimum:  # Avoid division by zero
        return np.full(vector.shape, lb)  # Return a constant vector

    return lb + ((vector - minimum) / (maximum - minimum)) * (ub - lb)

class Result():
    """
    Structure to store feature selection results.

    Attributes:
    - ranks: Feature ranking order.
    - scores: Computed feature scores.
    - features: Original feature data.
    - ranked_features: Features sorted by rank.
    """
    def __init__(self):
        self.ranks = None
        self.scores = None
        self.features = None
        self.ranked_features = None 

# -------------------------------------------
# Feature Selection Methods
# -------------------------------------------

#-------------------------  Chi-Square  -------------------------------------------#
@time_function
@profile_function(output_file=True)
def chi_square(data, target):
    """
    Computes the Chi-Square statistic for feature selection.

    Parameters:
    - data: Feature matrix (numpy array).
    - target: Class labels (numpy array).

    Returns:
    - result: `Result` object containing feature scores and rankings.
    """
    logger.info(f"Starting Chi-Square calculation with shape: {data.shape}")
    start_time = time.time()
    try:
        result = Result()
        logger.info("Clipping negative values...")
        result.features = np.array(pd.DataFrame(data).clip(lower=0))
        
        logger.info("Computing Chi-Square scores...")
        result.scores = chi2(result.features, target)[0]
        result.scores = normalize(result.scores)
        logger.info("Computing ranks...")
        result.ranks = np.argsort(np.argsort(-result.scores))
        result.ranked_features = result.features[:, result.ranks]
        
        total_time = time.time() - start_time
        logger.info(f"Chi-Square completed in {total_time:.2f} seconds")
        logger.info(f"Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")
        return result
    except Exception as e:
        logger.error(f"Error in Chi-Square calculation: {str(e)}", exc_info=True)
        raise

#-------------------------  Information Gain  -------------------------------------------#
@time_function
@profile_function(output_file=True)
def info_gain(data, target):
    """
    Computes the Information Gain (Mutual Information) for feature selection.

    Parameters:
    - data: Feature matrix (numpy array).
    - target: Class labels (numpy array).

    Returns:
    - result: `Result` object containing feature scores and rankings.
    """
    logger.info(f"Starting Information Gain calculation with shape: {data.shape}")
    start_time = time.time()
    try:
        result = Result()
        result.features = data
        
        logger.info("Computing mutual information scores...")
        result.scores = mutual_info_classif(data, target)
        result.scores = normalize(result.scores)
        logger.info("Computing ranks...")
        result.ranks = np.argsort(np.argsort(-result.scores))
        result.ranked_features = result.features[:, result.ranks]
        
        total_time = time.time() - start_time
        logger.info(f"Information Gain completed in {total_time:.2f} seconds")
        logger.info(f"Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")
        return result
    except Exception as e:
        logger.error(f"Error in Information Gain calculation: {str(e)}", exc_info=True)
        raise

#-------------------------  Mean Absolute Deviation  -------------------------------------------#
@time_function
@profile_function(output_file=True)
def MAD(data, target=None):
    """
    Computes the Mean Absolute Deviation (MAD) for feature selection.

    Parameters:
    - data: Feature matrix (numpy array).
    - target: (Optional) Class labels (numpy array).

    Returns:
    - result: `Result` object containing feature scores and rankings.
    """
    logger.info(f"Starting MAD calculation with shape: {data.shape}")
    start_time = time.time()
    try:
        result = Result()
        result.features = data
        
        logger.info("Computing mean values...")
        mean_vals = np.mean(data, axis=0)
        
        logger.info("Computing absolute deviations...")
        abs_devs = np.abs(data - mean_vals)
        
        logger.info("Computing MAD scores...")
        result.scores = np.mean(abs_devs, axis=0)
        result.scores = normalize(result.scores)
        logger.info("Computing ranks...")
        result.ranks = np.argsort(np.argsort(-result.scores))
        result.ranked_features = result.features[:, result.ranks]
        
        total_time = time.time() - start_time
        logger.info(f"MAD completed in {total_time:.2f} seconds")
        logger.info(f"Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")
        return result
    except Exception as e:
        logger.error(f"Error in MAD calculation: {str(e)}", exc_info=True)
        raise

#-------------------------  Dispersion Ratio  -------------------------------------------#
def Dispersion_ratio(data, target=None):
    """
    Computes the dispersion ratio for feature selection.
    """
    # Convert data to numpy array if it isn't already
    data = np.array(data, dtype=float)
    
    # Handle non-positive values (both zeros and negatives)
    min_positive = np.min(np.abs(data[data != 0])) if np.any(data != 0) else 1e-10
    data = np.where(data <= 0, min_positive, data)  # Replace zeros and negatives with small positive value
    
    result = Result()
    result.features = data
    
    am = np.mean(result.features, axis=0)  # Arithmetic mean
    gm = np.exp(np.mean(np.log(result.features), axis=0))  # Geometric mean
    
    result.scores = am / gm  # Compute dispersion ratio
    result.scores = normalize(result.scores)
    result.ranks = np.argsort(np.argsort(-result.scores))
    result.ranked_features = result.features[:, result.ranks]
    
    return result
#-------------------------  Pasi Luukka  -------------------------------------------#
@time_function
@profile_function(output_file=True)
def feature_selection_sim(in_data, target, measure='luca', p=1):
    """
    Implements Pasi Luukka's similarity-based feature selection.

    Parameters:
    - in_data: Feature matrix.
    - target: Class labels.
    - measure: Distance metric ('luca' for Luukka’s entropy, 'other' for sinusoidal).
    - p: Power factor for similarity computation.

    Returns:
    - result: `Result` object with feature scores and rankings.
    """
    logger.info(f"Starting Pasi Luukka calculation with shape: {in_data.shape}")
    start_time = time.time()
    try:
        # Convert data to DataFrame
        logger.info("Converting data to DataFrame...")
        d = pd.DataFrame(in_data)
        
        # Convert target to numeric
        logger.info("Converting target to numeric...")
        unique_labels = np.unique(target)
        target_numeric = np.zeros_like(target, dtype=float)
        for i, label in enumerate(unique_labels):
            target_numeric[target == label] = float(i + 1)
        
        logger.info(f"Found {len(unique_labels)} unique classes")
        
        # Combine data and target for processing
        t = pd.DataFrame(target_numeric)
        data = pd.concat([d, t], axis=1)
        
        l = len(unique_labels) # Number of unique classes
        m = data.shape[0] # Number of samples
        t = data.shape[1]-1 # Number of features
        
        logger.info("Computing ideal vectors...")
        
        # Compute ideal vectors per class
        idealvec_s = np.zeros((l,t))
        for k in range(l):
            idx = data.iloc[:,-1] == k+1
            idealvec_s[k,:] = data[idx].iloc[:,:-1].mean(axis = 0)
            logger.info(f"Processed class {k+1}/{l}")
        
        logger.info("Scaling data...")
        data_v = data.iloc[:,:-1]
        data_c = data.iloc[:,-1]
        
        # Rest of the scaling operations...
        logger.info("Computing similarities...")
        sim = np.zeros((t,m,l))
        total_computations = t * m * l
        completed = 0
        last_percentage = 0
        
        # Compute similarity for each feature and each class
        for j in range(m):
            for i in range(t):
                for k in range(l):
                    sim[i,j,k] = (1-abs(idealvec_s[k,i]**p - data_v.iloc[j,i])**p)**(1/p)
                    completed += 1
                    
                    # Log progress every 10%
                    percentage = (completed * 100) // total_computations
                    if percentage > last_percentage and percentage % 10 == 0:
                        logger.info(f"Similarity computation: {percentage}% complete")
                        last_percentage = percentage
        
        sim = sim.reshape(t,m*l)
        
        logger.info(f"Computing entropy using {measure} measure...")
        if measure == 'luca':
            delta = 1e-10
            sim[sim == 1] = delta
            sim[sim == 0] = 1-delta
            H = (-sim*np.log(sim)-(1-sim)*np.log(1-sim)).sum(axis = 1)
        else:
            H = (np.sin(np.pi/2*sim)+np.sin(np.pi/2*(1-sim))-1).sum(axis = 1)
        
        logger.info("Computing final results...")
        feature_values = np.array(in_data)
        result = Result()
        result.features = feature_values
        result.scores = H
        result.scores = normalize(result.scores)
        result.ranks = np.argsort(np.argsort(-H))
        result.ranked_features = feature_values[:, result.ranks]
        
        total_time = time.time() - start_time
        logger.info(f"Pasi Luukka completed in {total_time:.2f} seconds")
        logger.info(f"Score range: [{H.min():.4f}, {H.max():.4f}]")
        return result
        
    except Exception as e:
        logger.error(f"Error in Pasi Luukka calculation: {str(e)}", exc_info=True)
        raise

#-------------------------  Fisher Score  -------------------------------------------#
@time_function
@profile_function(output_file=True)
def Fisher_score(data, target):
    """
    Computes the Fisher Score for feature selection.
    
    Fisher Score evaluates the discriminative power of each feature by measuring 
    the ratio of between-class variance to within-class variance.

    Parameters:
    data (numpy array): Feature matrix of shape (n_samples, n_features)
    target (numpy array): Target labels of shape (n_samples,)

    Returns:
    result (Result object): Contains feature rankings, scores, and ranked feature matrix.
    """
    logger.info(f"Starting Fisher Score calculation with data shape: {data.shape}")
    start_time = time.time()
    
    try:
        result = Result()
        result.features = data
        
        # Convert and reshape data
        data = np.array(data, dtype=np.float64)
        target = np.array(target).ravel()
        
        logger.info("Computing class statistics...")
        class_start = time.time()
        
        # Compute global mean once
        global_mean = np.mean(data, axis=0)
        classes = np.unique(target)
        n_classes = len(classes)
        
        # Pre-compute class masks
        class_masks = np.array([target == c for c in classes])
        logger.info(f"Found {n_classes} classes")
        
        # Compute class means and variances
        class_means = np.array([np.mean(data[mask], axis=0) for mask in class_masks])
        class_vars = np.array([np.var(data[mask], axis=0) for mask in class_masks])
        class_sizes = np.array([np.sum(mask) for mask in class_masks])
        
        class_time = time.time() - class_start
        logger.info(f"Class statistics computed in {class_time:.2f} seconds")
        
        # Compute Fisher scores
        logger.info("Computing Fisher scores...")
        score_start = time.time()
        
        numerator = np.sum(class_sizes[:, None] * (class_means - global_mean) ** 2, axis=0)
        denominator = np.sum(class_sizes[:, None] * class_vars, axis=0)
        denominator = np.where(denominator < 1e-10, np.inf, denominator)
        fisher = numerator / denominator
        
        score_time = time.time() - score_start
        logger.info(f"Scores computed in {score_time:.2f} seconds")
        
        result.scores = fisher
        result.scores = normalize(result.scores)
        result.ranks = np.argsort(np.argsort(-fisher))
        result.ranked_features = result.features[:, result.ranks]
        
        total_time = time.time() - start_time
        logger.info(f"Fisher Score calculation completed in {total_time:.2f} seconds")
        logger.info(f"Score range: [{fisher.min():.4f}, {fisher.max():.4f}]")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Fisher Score calculation: {str(e)}", exc_info=True)
        logger.error(f"Data shape: {data.shape}, Target shape: {target.shape}")
        raise

#-------------------ER Filter-------------------#
def ER_filter(X, y, gamma=1.732):
    """
    A filter method that uses Effective Range (ER) logic to assign one numeric score
    per feature, then ranks them. 

    Parameters
    ----------
    X : ndarray or DataFrame of shape (n_samples, n_features)
        Feature matrix
    y : 1D ndarray or Series of shape (n_samples,)
        Class labels (any type)
    gamma : float, default=1.732
        Factor for computing lower/upper bounds based on Chebyshev inequality

    Returns
    -------
    result : Result
        Contains:
          - result.features          -> the original X
          - result.scores            -> 1D array of scores (size = n_features)
          - result.ranks             -> rank array (best=0 if highest score is best)
          - result.ranked_features   -> columns of X in descending-score order
    """
    # Convert X,y to DataFrame/Series if needed
    X_df = pd.DataFrame(X)
    y_s  = pd.Series(y)

    # Collect lower/upper bounds for each (feature, class)
    rows = []
    for feat_idx, feat_name in enumerate(X_df.columns):
        for cls_val in np.unique(y_s):
            class_mask = (y_s == cls_val)
            mean_val   = X_df.loc[class_mask, feat_name].mean()
            std_val    = X_df.loc[class_mask, feat_name].std()
            Pj         = class_mask.sum() / len(y_s)

            lower_r = mean_val - (1 - Pj) * gamma * std_val
            upper_r = mean_val + (1 - Pj) * gamma * std_val

            rows.append({
                "feature_idx": feat_idx,
                "feature_name": feat_name,
                "class": cls_val,
                "lower_bound": lower_r,
                "upper_bound": upper_r,
                "effective_range": (upper_r - lower_r),
            })

    # Build a DataFrame of these ranges
    df_ranges = pd.DataFrame(rows)
    feature_scores = df_ranges.groupby("feature_idx")["effective_range"].mean()

    # Convert to a NumPy array, shape = (num_features,)
    scores = feature_scores.to_numpy()
    normalized_scores = normalize(scores)  # Normalize first
    ranks = np.argsort(np.argsort(-normalized_scores))

    # Build the Result object (same pattern as MI, info_gain, etc.)
    result = Result()
    result.features = X
    result.scores   = normalized_scores
    result.ranks    = ranks

    order = np.argsort(-normalized_scores)
    # Reorder columns in X (axis=1) by descending score
    result.ranked_features = X[:, order] if isinstance(X, np.ndarray) else X_df.iloc[:, order].values

    return result


#-------------------------  Relief  -------------------------------------------#
@time_function
@profile_function(output_file=True)
def Relief(data, target):
    logger.info(f"Starting Relief calculation with shape: {data.shape}")
    start_time = time.time()
    
    try:
        feature_values = data
        num_features = feature_values.shape[1]
        result = Result()
        result.features = feature_values

        # Ensure data is in the correct format
        logger.info("Converting data to correct format...")
        data = np.array(data, dtype=float)
        target = np.array(target).reshape(-1)

        try:
            logger.info("Attempting ReliefF calculation...")
            relief = ReliefF(n_neighbors=5, n_features_to_keep=num_features)
            relief.fit(data, target)
            scores = relief.feature_scores
            logger.info("ReliefF calculation successful")
        except ValueError as e:
            logger.warning(f"ReliefF failed: {str(e)}. Using fallback implementation...")
            scores = np.zeros(num_features)
            total_features = num_features
            processed = 0
            
            for i in range(num_features):
                feature = data[:, i].reshape(-1)
                if np.std(feature) == 0 or np.std(target) == 0:
                    scores[i] = 0
                else:
                    scores[i] = abs(np.corrcoef(feature, target)[0, 1])
                
                processed += 1
                if processed % max(1, total_features // 10) == 0:
                    logger.info(f"Processed {processed}/{total_features} features ({(processed/total_features)*100:.1f}%)")
        
        logger.info("Computing final scores and ranks...")
        result.scores = normalize(scores)
        result.ranks = np.argsort(np.argsort(-scores))
        result.ranked_features = feature_values[:, result.ranks]
        
        total_time = time.time() - start_time
        logger.info(f"Relief completed in {total_time:.2f} seconds")
        logger.info(f"Score range: [{result.scores.min():.4f}, {result.scores.max():.4f}]")
        return result
        
    except Exception as e:
        logger.error(f"Error in Relief calculation: {str(e)}", exc_info=True)
        raise

#-------------------------  Spearman's Correlation Coefficient  -------------------------------------------#
def SCC(data, target):
    """
    Computes Spearman’s Rank Correlation Coefficient (SCC) for feature selection.

    Parameters:
    - data: Feature matrix.
    - target: Class labels.

    Returns:
    - result: `Result` object containing feature scores and rankings.
    """
    result = Result()
    result.features = data
    num_features = data.shape[1]
    
    # Use pandas' optimized correlation computation
    SCC_mat = pd.DataFrame(data).corr(method="spearman").values
    
    # Vectorized computation of feature-feature correlations
    SCC_values_feat = -np.sum(np.abs(SCC_mat), axis=1)
    
    # Vectorized computation of feature-class correlations
    SCC_values_class = np.array([spearmanr(data[:, i], target)[0] for i in range(num_features)])
    
    # Replace NaN values with 0
    SCC_values_class = np.nan_to_num(SCC_values_class)
    
    # Normalize and compute final scores
    SCC_values_feat = normalize(SCC_values_feat)
    SCC_values_class = normalize(SCC_values_class)
    SCC_scores = 0.7 * SCC_values_class + 0.3 * SCC_values_feat
    
    result.scores = SCC_scores
    result.ranks = np.argsort(np.argsort(-SCC_scores))
    result.ranked_features = result.features[:, np.argsort(-SCC_scores)]
    
    return result

#-------------------------  ANOVA F-Value  -------------------------------------------#
def ANOVA_F(data, target):
    """
    Computes ANOVA F-value for feature selection.

    Parameters:
    - data: Feature matrix (numpy array).
    - target: Class labels (numpy array).

    Returns:
    - result: `Result` object containing feature scores and rankings.
    """

    result = Result()
    result.features = data

    # Compute ANOVA F-value scores
    f_values, _ = f_classif(data, target)

    # Compute ranks based on scores
    result.scores = f_values
    result.scores = normalize(result.scores)
    result.ranks = np.argsort(np.argsort(-f_values))
    result.ranked_features = result.features[:, np.argsort(-f_values)]

    return result