"""main.py - for the milestone update 11/23/26

Loads the data cleaned in Stata, splits it by year, fits a dummy baseline and a first decision tree on 2010-2020, and scores both on 2010-2020 and on 2022+2024

Sources: Geron (2022)
    Ch. 2: SimpleImputer, OneHotEncoder, ColumnTransformer, make_pipeline
    Ch. 3: DummyClassifier, confusion matrix, accuracy, precision, recall, F1
    Ch. 6: DecisionTreeClassifier 

AI Disclaimer: While I did not use generative AI to generate any code, I used Claude and/or Google AI to:
    1: Guide me to sections of Geron (2022) that could be helpful in building the decision tree;
    2: Answer to any confusion I had regarding specific concepts or packages - which packages to download;
    3: Explain how decision trees are used in applied GOTV campaigns - i.e., general guidance to specific variables and observational n requirements; and 
    4: Explain errors so I could debug my code independently 

"""

# Below are the imports - these load the tools from pandas and scikit learns
    # google AI helped me understand what each of these imports were doing specifically

from pathlib import Path # file and folder paths
import pandas as pd # load in pandas
from sklearn.compose import ColumnTransformer # imports ColumnTransformer to apply different preprocessing steps to specific subsets of columns simultaneously
from sklearn.dummy import DummyClassifier # the baseline model which ignores input features
from sklearn.impute import SimpleImputer # fills in missing values with summary statistics (median)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score # ch. 3 assessment scores
from sklearn.pipeline import make_pipeline # chains multiple processes into a single process
from sklearn.preprocessing import OneHotEncoder # turns one text column into several 0/1 columns
from sklearn.tree import DecisionTreeClassifier # imports decision tree model

# Below state which files to read and which variables are numbers and which are text
# The names of the variables match those that I operationalized directly in Stata (See clean_data.do for how I cleaned the files)

DATA = "data/sample/ces_clean_sample.csv" # this is just a sample of the full data, for minimal examination - for the full dataset (400k+ observations), see data/clean/ces_clean_full.csv)
NUMERIC = [ "age", "educ", "income", "news_interest", "voted_primary", "voted_prior_pres"] # features that are already numeric
CATEGORICAL = ["race_group", "housing", "party", "state"] # categorical features that I will make numeric/binary

# Below I load and split the data by year
# the model learns from 2010-2020 and is tested on 2022 and 2024 (elections it never saw)
# note: x holds the features (IV in poli sci terms), and y holds the answer (DV in poli sci terms (voted))

# First, I load the file that I made using clean_data.do; learn from 2010-2020, test on 2022-24
df = pd.read_csv(DATA) # reads my CSV file w/ the sample data
train = df[df["year"] <= 2020] # train only on 2010-2020
test = df[df["year"] >= 2022] # test only on 2022-2024
X_train, y_train = train[NUMERIC + CATEGORICAL], train["voted"] # features (x) as all the numeric/categorical variables above; (y) as binary "voted" variable 
X_test, y_test = test[NUMERIC + CATEGORICAL], test["voted"] # same split as above for the test election years

# Below fills in missing values from stata and turns each text variable into binary 0/1 variables
# B/c it runs inside the pipeline, it should only learn from the training years

prep = ColumnTransformer([ # specifying one recipe for each group of columns (numeric or categorical)
    ("num", SimpleImputer(strategy="median"), NUMERIC),  # for numeric features: impute blanks with training median values
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),  # for categorical features: impute blanks with most frequent value
                          OneHotEncoder(handle_unknown="ignore")), CATEGORICAL), # converts categorical values into 0/1 columns; makes sure a category never seen in training gets all 0s
])

# quick check while I push 
print(len(train), len(test)) # prints the number of training and test rows
    # 12000 4000
    # meaning 2k people x 6 training years and 2k x 2 test years
    # import successful!

# Below creates the first models
# The dummy should always guess "voted"
# The first tree has no size limits FOR NOW ** will adjust after milestone update ** 
# This is the baseline and the first tree

models = {
    "dummy": DummyClassifier(strategy="most_frequent"), # the baseline dummy predicts that everyone votes
    "tree_one": make_pipeline(prep, DecisionTreeClassifier(random_state=42)), # the first tree: no size limits yet; random set at 42 for replicability 
}

# Below fits and scores
# Each model trains on 2010-20, then predicts both the training years and the test years
# The for loop records the four (Ch. 3) scores and prints the confusion matrix 

rows = [] # empty list; one row of scores is added per model per dataset
for name, model in models.items(): # basic for loop; used Claude AI to help with the basic structure of a for loop in python
    model.fit(X_train, y_train) # train on 2010-2020 elections (for the tree: prep learns medians/categories, then the tree learns its splits)
    for data_name, X, y in [("train 2010-2020", X_train, y_train), ("test 2022+2024", X_test, y_test)]: # score on training, then on test
        pred = model.predict(X) # prediction step... voted (1) or not (0) for each individual respondent 
        rows.append({"model": name, "data": data_name, # append rows of scores) 
                     "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred), # print accuracy score: share of correct guesses (i.e., of predicted voters, share who voted))
                     "recall": recall_score(y, pred), "f1": f1_score(y, pred)}) # print recall scores: of actual voters, share the model caught; print F1: combines precision and recall 
    print(f"{name} confusion matrix on test [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, model.predict(X_test))}") # prints confusion matrix

# print and save the general scores
    # google AI helped guide me on the below code - this was my first time saving results in Python

results = pd.DataFrame(rows).round(3) # presents results in a table (rounded to 3 decimal places)
print(results.to_string(index=False)) # print said table
Path("results").mkdir(exist_ok=True) # make the results folder
results.to_csv(f"results/metrics_{Path(DATA).stem}.csv", index=False) # save results in folder

    # note to self: after each edit, 
    # python main.py
    # git add main.py
    # git commmit -m "explain what I changed"
    # git push