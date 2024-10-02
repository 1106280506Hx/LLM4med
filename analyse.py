from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

from pathlib import Path

import cv2
import mediapy
import pandas as pd
import plotly.express as px
import pycolmap

import numpy as np # linear algebra
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input/imc2024-packages-lightglue-rerun-kornia'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_labels = pd.read_csv("/kaggle/input/image-matching-challenge-2024/train/train_labels.csv")
train_labels

train_labels.groupby("dataset")["scene"].nunique()

dataset_counts = train_labels["dataset"].value_counts()

fig = px.pie(values=dataset_counts.values, names=dataset_counts.index)
fig.update_traces(textposition='inside', textfont_size=14)
fig.update_layout(
    title={
        'text': "Pie distribution of dataset images",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend_title_text='Dataset names:'
)
fig.show()

train_categories = pd.read_csv("/kaggle/input/image-matching-challenge-2024/train/categories.csv")

# From comma separated list of categories for each dataset
# To one dataset & category per row
train_categories["category"] = train_categories["categories"].str.split(";")
train_categories = train_categories.explode("category")

category_counts = train_categories["category"].value_counts()

fig = px.pie(values=dataset_counts.values, names=category_counts.index)
fig.update_traces(textposition='inside', textfont_size=14)
fig.update_layout(
    title={
        'text': "Pie distribution of categories of datasets",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend_title_text='Category names:'
)
fig.show()

fig = px.sunburst(train_categories, path=['scene', 'category'])
fig.update_layout(
    title={
        'text': "Scene and category relation",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

fig.show()

def explore(split: str, dataset: str, plot_image_limit: int = 12) -> None:
    path = Path("/kaggle/input/image-matching-challenge-2024") / split / dataset
    images_path = path / "images"
    smf_path = path / "sfm"    

    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in list(images_path.glob("*"))[:plot_image_limit]]
    mediapy.show_images(images, height=300, columns=3)
    
    if split != "test":
        rec_gt = pycolmap.Reconstruction(smf_path)

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, rec_gt, cameras=False, color='rgba(227,168,30,0.5)', name="Ground Truth", cs=5)
        fig.show()

explore(split="train", dataset="church")