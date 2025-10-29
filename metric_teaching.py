"""
Copyright (c) 2024 Tom Koller
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial.distance import directed_hausdorff, cdist
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


def update_canvas(x, y, annotation, value):
    """Update the canvas with a brush stroke at (x, y)."""
    half_brush = brush_size // 2
    x_min, x_max = max(0, x - half_brush), min(canvas_size[0], x + half_brush + 1)
    y_min, y_max = max(0, y - half_brush), min(canvas_size[1], y + half_brush + 1)
    annotation[x_min:x_max, y_min:y_max,0] = 1


def recomputeMetrics():
    metrics =[function(reference_annotation, ml_example_annotation) for function in metricFunctions]
    metrics = [y for x in metrics for y in ((x,) if isinstance(x, float) else x)] ## Flatten since some function output multiple metrics
    for i, val in enumerate(metrics):
        table[i+1, 0].get_text().set_text(f'{val:.2f}')  # Update text in each cell

def clear():
    reference_annotation[:,:,:] = 0
    ml_example_annotation[:,:,:] = 0
    redraw()

def on_key(event):
    if event.key == "escape":
        clear()

def draw(buttonNumber, x, y):
    if buttonNumber == 1:
        update_canvas(x, y, reference_annotation, 1)
    elif buttonNumber == 3:
        update_canvas(x, y, ml_example_annotation, 1)
    redraw()

def on_click(event):
    global lastButton
    """Handle mouse click events."""
    if event.inaxes != ax[0]:
        return
    x, y = int(event.ydata), int(event.xdata)
    draw(event.button, x, y)
    lastButton = event.button
    
def on_release(event):
    global lastButton
    if event.inaxes != ax[0]:
        return
    lastButton = -1
    recomputeMetrics()
    redraw()

def on_move(event):
    if event.inaxes != ax[0]:
        return
    if lastButton != -1:
        x, y = int(event.ydata), int(event.xdata)
        draw(lastButton, x,y)

def updateBrushSize(val):
    global brush_size
    brush_size = val

def redraw():
    """Redraw the updated canvas."""
    rgbImage = np.concatenate([canvas, reference_annotation, ml_example_annotation], axis = -1)
    shownImage.set_data(rgbImage)
    fig.canvas.draw()
    fig.canvas.flush_events()


##metric functions

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


def compute_confusion_matrix(reference, prediction):
    # Flatten the 2D images to 1D arrays
    if np.max(reference) == 0 or np.max(prediction) == 0:
        return [0, np.sum(prediction), np.sum(reference) , len(reference.flatten())- np.sum(reference)-np.sum(prediction) ]
    reference_flat = reference.flatten()
    prediction_flat = prediction.flatten()
    # Compute the confusion matrix
    cm = confusion_matrix(reference_flat,prediction_flat)
    tn, fp, fn, tp = cm.ravel()
    return [tp, fp, fn, tn]


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





# Create an empty canvas
canvas_size = (100, 100, 1)
canvas = np.zeros(canvas_size)

# Initialize annotations
reference_annotation = np.zeros_like(canvas)
ml_example_annotation = np.zeros_like(canvas)

# Set up the plot
fig, ax = plt.subplots(1,2)
rgbImage = np.concatenate([canvas, reference_annotation, ml_example_annotation], axis = -1)
shownImage = ax[0].imshow(rgbImage, vmin=0, vmax=1)
brush_size = 9
lastButton = -1

ax_freq = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax=ax_freq, label='brush_size', valmin=1, valmax=31, valinit=brush_size, valstep = 2)
slider.on_changed(updateBrushSize)


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
# Flatten metricNames in Tuples
metricNames = list(metricDict.keys())
metricNames = [y for x in metricNames for y in (x if isinstance(x, tuple) else (x,))]

metricFunctions = metricDict.values()

metrics = [[1.0] for entry in metricNames]
ax[1].axis('tight')
ax[1].axis('off')
table = ax[1].table(cellText=metrics, colLabels=["Value"], rowLabels=metricNames, loc='center')

table.auto_set_column_width(col = 0)

plt.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.2)



# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()