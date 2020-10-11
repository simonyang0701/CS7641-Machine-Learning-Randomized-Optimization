# CS 7641 Assignment 2: Randomized Optimization
## Author
Name: Tianyu Yang<br>
GTid: 903645962<br>
Date: 2020/10/11<br>

## Introduction
This project is to generate four optimization algorithms(Randomized Hill Climbing, Simulated Annealing, Genetic Algorithm and MIMC) on three problems, which is Four Peaks, One Max and Knapsack. Besides, three of the optimization algorithms are used in the Neural Network model I built in Assignment 1: Supervised Learning problem. The dataset I used is downloaded from Kaggle<br>
NBA games: https://www.kaggle.com/nathanlauga/nba-games<br>

This project link on Github: https://github.com/simonyang0701/CS7641-Machine-Learning-Randomized-Optimization.git<br>


## Getting Started & Prerequisites
To test the code, you need to make sure that your python 3.6 is in recent update and the following packages have already been install:
pandas, numpy, scikit-learn, matplotlib, itertools, timeit

mlrose for machine learning optimization algorithm package has already in the repository so that you do not need to import again.


## Running the Classifiers
Recommendation Option: Work with the iPython notebook (.ipnyb) using Jupyter or a similar environment. Use "Run ALL" in Cell to run the code. Before running the code, make sure that you have already change the path into your current working directory
Another Option: Run the python script (.py) after changing the directory into where you saved the two datasets
Other Option (view only): Feel free to open up the (.html) file to see the output of program

The Optimization Algorithms are divided into three parts:
1. Importing useful packages
2. Build up the problem and get the optimization result
3. Plot the result in figure

The Neural Network codes are divided into four parts:
1. Importing useful packages
2. Loading and cleaning datasets: load and clean the datasets
3. Running two Neural Networks models and get the training and testing time
4. Run three optimization algorithm with the same Neural Networks model
