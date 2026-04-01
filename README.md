# Boston House Price Prediction

# Software and tools required

1. [Github Account](https://github.com)
2. [Heroku Account](https://heroku.com)
3. [VS Code IDE](https://code.visualstudio.com/download)
4. [Git CLI](https://git-scm.com/downloads)

This project aims to predict the median value of owner-occupied homes in Boston using machine learning techniques. The dataset used is the famous Boston Housing Dataset, which contains information about various factors that might influence housing prices in the Boston area.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Boston Housing Dataset is a well-known dataset in the machine learning community. It contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts. The dataset includes 506 samples and 13 feature variables, such as crime rate, number of rooms, accessibility to radial highways, and more. The target variable is the median value of owner-occupied homes in $1000s.

## Dataset

The dataset used in this project is the Boston Housing Dataset, which can be loaded using the `load_boston` function from the `sklearn.datasets` module. The dataset contains the following


Create a new environment
```bash
conda create -p venv python==3.7 -y
```
Activate the environment
```bash
conda activate venv/
```
Install the requirements
```bash
pip install -r requirements.txt
