"""
Copyright (c) 2024 Tom Koller
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

#Wrappers

def wrap_kwargs(func, **fixed_kwargs):
    """Use to set args from functions without flattening"""
    def wrapper(y_ref, y_pred, *args, **kwargs):
        merged = {**fixed_kwargs, **kwargs}
        return func(y_ref, y_pred, *args, **merged)
    return wrapper

def flatten_inputs(func, **fixed_kwargs):
    def wrapper(y_ref, y_pred, *args, **kwargs):
        merged = {**fixed_kwargs, **kwargs}
        # Flatten the input images
        return func(np.ravel(y_ref), np.ravel(y_pred), *args, **merged)
    return wrapper

# Metrics 

def dice_score(image1, image2):
    if np.max(image1) == 0 and np.max(image2) == 0:
        return [-1]
    # Ensure the images are binary (0 or 1)
    image1 = image1.astype(bool)
    image2 = image2.astype(bool)
    # Calculate intersection and sum
    intersection = np.logical_and(image1, image2)
    dice = 2. * intersection.sum() / (image1.sum() + image2.sum())
    return [dice]

def jaccard_index(image1, image2):
    if np.max(image1) == 0 and np.max(image2) == 0:
        return [-1]
    # Ensure the images are binary (0 or 1)
    image1 = image1.astype(bool)
    image2 = image2.astype(bool)
    # Compute intersection and union
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1, image2).sum()
    # Calculate Jaccard index
    jaccard = intersection / union
    return [jaccard]

def hausdorff_distance(image1, image2):
    if np.max(image1) == 0 or np.max(image2) == 0:
        return [np.nan]
    # Retrieve the non-zero points (coordinates) in each image
    points1 = np.argwhere(image1)
    points2 = np.argwhere(image2)
    # Compute directed Hausdorff distances between the point sets
    dist1 = directed_hausdorff(points1, points2)[0]
    dist2 = directed_hausdorff(points2, points1)[0]
    # The Hausdorff distance is the maximum of the two directed distances
    hausdorff = max(dist1, dist2)
    return [hausdorff]

def hausdorff_percentile(image1, image2, percentile=95):
    if np.max(image1) == 0 or np.max(image2) == 0:
        return [np.nan]
    # Retrieve the non-zero points (coordinates) in each image
    points1 = np.argwhere(image1)
    points2 = np.argwhere(image2)
    # Compute all pairwise distances from points1 to points2
    distances = cdist(points1, points2)
    # For each point in points1, find the minimum distance to points2
    min_distances = distances.min(axis=1)
    # Compute the specified percentile of these minimum distances
    percentile_value = np.percentile(min_distances, percentile)
    return [percentile_value]

@flatten_inputs
def compute_confusion_matrix(reference, prediction):
    if np.max(reference) == 0 or np.max(prediction) == 0:
        return [0, np.sum(prediction), np.sum(reference) , len(reference)- np.sum(reference)-np.sum(prediction) ]
    # Compute the confusion matrix
    cm = confusion_matrix(reference,prediction)
    tn, fp, fn, tp = cm.ravel()
    return [tp, fp, fn, tn]





# Declare Metrics (Use Tuple to have multiple outputs of a function)
metricDict = {
    ("TP", "FP", "FN", "TN") : compute_confusion_matrix, 
    "Accuracy"    : flatten_inputs(accuracy_score),
    "Sensitivity" : flatten_inputs(recall_score, zero_division = np.nan), 
    "Specificity" : flatten_inputs(recall_score, pos_label=0, zero_division = np.nan),
    "Precision"   : flatten_inputs(precision_score, zero_division = np.nan),
    "NPV"         : flatten_inputs(precision_score, pos_label=0, zero_division = np.nan),
    "DICE"        : dice_score, 
    "IoU"         : jaccard_index,
    "HD"          : hausdorff_distance,
    "HD95"        : hausdorff_percentile
}
