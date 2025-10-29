"""
Copyright (c) 2024 Tom Koller
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from metric_functions import *


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