
import plotly.graph_objects as go
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def bar_chart(labels, sizes):
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])
    fig.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Churn', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='No Churn', x=0.87, y=0.5, font_size=20, showarrow=False)])
    return fig

def confusion_matrix(y_true, y_pred):
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion matrix", y=1.1)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    return fig