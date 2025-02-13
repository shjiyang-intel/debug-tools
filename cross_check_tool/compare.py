import numpy as np

# np.set_printoptions(threshold=np.inf)
THRESHOLDS = [5, 5, 5, 5, 5]
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
NORMAL = '\033[0m'
BOLD = '\033[1m'
PURPLE = '\033[95m'

def metrics(a, b):
    ref = np.max(np.abs(b))
    total_values = int(len(b.flatten()))
    diff = np.abs(a - b)
    max_error = np.max(np.abs(a - b))
    mean_error = np.mean(np.abs(a - b))
    l2_error = np.sqrt(np.sum(np.square(a - b)) / total_values)
    # print(np.max(a))
    # print(np.max(b))
    if ref == 0:
        max_error = 0 if max_error == 0 else np.inf
        mean_error = 0 if mean_error == 0 else np.inf
        l2_error = 0 if l2_error == 0 else np.inf
    else:
        max_error = max_error / ref * 100
        mean_error = mean_error / ref * 100
        l2_error = l2_error / ref * 100
    percentage_wrong = len(
        np.extract(
            diff > 0.05 * ref,
            diff)) / total_values * 100
    #sum_diff = np.sum(np.abs(a - b))
    sum_diff = np.sum(diff) / np.sum(np.abs(b)) * 100
    return [max_error, mean_error, percentage_wrong, l2_error, sum_diff]

def metrix_comparison(results):
    status = []
    for i in range(5):
        if results[i] > THRESHOLDS[i] or np.isnan(results[0]).any():
            status.append(FAIL + "Fail" + NORMAL)
        else:
            status.append("Pass")
    print("------------------------------------------------------------")
    print(" Obtained values ")
    print("------------------------------------------------------------")
    print(
        " Obtained Percentage of wrong values: {}% (max allowed={}%), {}".format(
            results[2],
            THRESHOLDS[2],
            status[2]))
    print(
        " Obtained Global Sum Difference values: {}% (max allowed={}%), {}".format(
            results[4],
            THRESHOLDS[4],
            status[4]))
    
    print(
        " Obtained Min Pixel Accuracy: {}% (max allowed={}%), {}".format(
            results[0],
            THRESHOLDS[0],
            status[0]))
    print(
        " Obtained Average Pixel Accuracy: {}% (max allowed={}%), {}".format(
            results[1],
            THRESHOLDS[1],
            status[1]))
    print(
        " Obtained Pixel-wise L2 error: {}% (max allowed={}%), {}".format(
            results[3],
            THRESHOLDS[3],
            status[3]))
          
    print("------------------------------------------------------------")
    return status[0] == 'Pass' and status[1] == 'Pass' and status[2] == 'Pass' and status[3] == 'Pass'

def metrix_compare(act, ref):
        metrix_comparison(metrics(act, ref))