"""main.py - for the milestone update 11/23/26

Loads the data cleaned in Stata, splits it by year, fits a dummy baseline and a first decision tree on 2010-2020, and scores both on 2010-2020 and on 2022+2024

Sources: Geron (2022)
    Ch. 2: SimpleImputer, OneHotEncoder, ColumnTransformer, make_pipeline
    Ch. 3: DummyClassifier, confusion matrix, accuracy, precision, recall, F1
    Ch. 6: DecisionTreeClassifier 

AI Disclaimer: While I did not use generative AI to generate any code, I used Claude to:
    1: Guide me to sections of Geron (2022) that could be helpful in building the decision tree;
    2: Answer to any confusion I had regarding specific concepts or packages - which packages to download;
    3: Explain how decision trees are used in applied GOTV campaigns - i.e., general guidance to specific variables and observational n requirements; and 
    4: Explain errors so I could debug my code independently 

"""

# Below are the imports - these load the tools from pandas and scikit learns

from pathlib import Path 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# Below state which files to read and which variables are numbers and which are text
# The names of the variables match those that I operationalized directlty in Stata (See clean_data.do for how I cleaned the files)

DATA = "data/sample/ces_clean_sample.csv" # this is just a sample of the full data, for minimal examiniation - for the full dataset (400k+ observations), see data/clean/ces_clean_full.csv)
NUMERIC = [ "age", "educ", "income", "news_interest", "voted_primary", "voted_prior_pres"]
CATEGORICAL = ["race_group", "housing", "party", "state"]

# Below I load and split the data by year
# the model learns from 2010-2020 and is tested on 2022 and 2024 (election returns it never saw)
# note: x holds the features (IV in poli sci terms), and y holds the answer (DV in poli sci terms (voted))

# First, I load the file that I made using clean_data.do; learn from 2010-2020, test on 2022-24
df = pd.read_csv(DATA)
train = df[df["year"] <= 2020]
test = df[df["year"] >= 2022]
X_train, y_train = train[NUMERIC + CATEGORICAL], train["voted"]
X_test, y_test = test[NUMERIC + CATEGORICAL], test["voted"]

# Below fills in missing values from stata and turns each text variable into binary 0/1 variables
# B/c it runs inside the pipeline, it should only learn from the training years

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUMERIC), 
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), 
                          OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
])

# quick check while I push 
print(len(train), len(test))
    # 12000 4000
    # meaning 2k people x 6 training years and 2k x 2 test years
    # import successful!

# Below creates the first models
# The dumy should always guess "voted"
# The first tree has no size limits FOR NOW
# This is the baseline and the first tree

models = {
    "dummy": DummyClassifier(strategy="most_frequent"), # the baseline predicts that everyone votes
    "tree_one": make_pipeline(prep, DecisionTreeClassifier(random_state=42)), # the first tree: no size limits yet
}

# Below fits and scores
# Each model trains on 2010-20, then predicts both the training years and the test years
# The for loop records the four (Ch. 3) scores and prints the confusion matrix 

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    for data_name, X, y in [("train 2010-2020", X_train, y_train), ("test 2022+2024", X_test, y_test)]:
        pred = model.predict(X)
        rows.append({"model": name, "data": data_name,
                     "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
                     "recall": recall_score(y, pred), "f1": f1_score(y, pred)})
    print(f"{name} confusion matrix on test [[TN FP] [FN TP]]:\n{confusion_matrix(y_test, model.predict(X_test))}")

# print and save the general scores

results = pd.DataFrame(rows).round(3)
print(results.to_string(index=False))
Path("results").mkdir(exist_ok=True)
results.to_csv(f"results/metrics_{Path(DATA).stem}.csv", index=False)