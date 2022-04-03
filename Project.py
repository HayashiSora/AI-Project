import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

import seaborn as sns; sns.set()

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from IPython.display import Image  
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans 

from PIL import Image

# Start
st.title("Question 3")
st.write("Study the dataset given carefully and build TWO classification and ONE cluster analysis programs in Python. You need to modify the parameters to achieve the highest accuracy.")


# Dataset
bank = pd.read_csv('Bank_CreditScoring.csv')


# Value to predict
X = bank.drop(['Employment_Type', 'More_Than_One_Products', 'Property_Type', 'State', 'Decision'], axis=1)


# Object
objInDataSet = ['Employment_Type', 'More_Than_One_Products', 'Property_Type', 'State', 'Decision']
obj = st.sidebar.selectbox(
    "Which option to do the investigate?",
    objInDataSet
)
y = bank[obj]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_predNb = nb.predict(X_test)


# Values
Credit_Card_Exceed_Months = st.sidebar.slider(
    "Credit Card Exceed Month",
    int(X['Credit_Card_Exceed_Months'].min()), int(X['Credit_Card_Exceed_Months'].max()),
    value=np.int(X['Credit_Card_Exceed_Months'].mean())
)

Loan_Amount = st.sidebar.slider(
    "Loan Amount",
    int(X['Loan_Amount'].min()), int(X['Loan_Amount'].max()),
    value=np.int(X['Loan_Amount'].mean())
)

Loan_Tenure_Year = st.sidebar.slider(
    "Loan Amount",
    int(X['Loan_Tenure_Year'].min()), int(X['Loan_Tenure_Year'].max()),
    value=np.int(X['Loan_Tenure_Year'].mean())
)

Credit_Card_More_Than_Months = st.sidebar.slider(
    "Credit Card More Than Months",
    int(X['Credit_Card_More_Than_Months'].min()), int(X['Credit_Card_More_Than_Months'].max()),
    value=np.int(X['Credit_Card_More_Than_Months'].mean())
)

Number_of_Dependents = st.sidebar.slider(
    "Number of Dependents",
    int(X['Number_of_Dependents'].min()), int(X['Number_of_Dependents'].max()),
    value=np.int(X['Number_of_Dependents'].mean())
)

Years_to_Financial_Freedom = st.sidebar.slider(
    "Years to Financial Freedom",
    int(X['Years_to_Financial_Freedom'].min()), int(X['Years_to_Financial_Freedom'].max()),
    value=np.int(X['Years_to_Financial_Freedom'].mean())
)

Number_of_Credit_Card_Facility = st.sidebar.slider(
    "Number of Credit Card Facility",
    int(X['Number_of_Credit_Card_Facility'].min()), int(X['Number_of_Credit_Card_Facility'].max()),
    value=np.int(X['Number_of_Credit_Card_Facility'].mean())
)

Number_of_Properties = st.sidebar.slider(
    "Number of Properties",
    int(X['Number_of_Properties'].min()), int(X['Number_of_Properties'].max()),
    value=np.int(X['Number_of_Properties'].mean())
)

Number_of_Bank_Products = st.sidebar.slider(
    "Number of Bank_Products",
    int(X['Number_of_Bank_Products'].min()), int(X['Number_of_Bank_Products'].max()),
    value=np.int(X['Number_of_Bank_Products'].mean())
)

Number_of_Loan_to_Approve = st.sidebar.slider(
    "Number of Loan to Approve",
    int(X['Number_of_Loan_to_Approve'].min()), int(X['Number_of_Loan_to_Approve'].max()),
    value=np.int(X['Number_of_Loan_to_Approve'].mean())
)

Years_for_Property_to_Completion = st.sidebar.slider(
    "Years for Property to Completion",
    int(X['Years_for_Property_to_Completion'].min()), int(X['Years_for_Property_to_Completion'].max()),
    value=np.int(X['Years_for_Property_to_Completion'].mean())
)

Number_of_Side_Income = st.sidebar.slider(
    "Number of Side Income",
    int(X['Number_of_Side_Income'].min()), int(X['Number_of_Side_Income'].max()),
    value=np.int(X['Number_of_Side_Income'].mean())
)

Monthly_Salary = st.sidebar.slider(
    "Monthly Salary",
    int(X['Monthly_Salary'].min()), int(X['Monthly_Salary'].max()),
    value=np.int(X['Monthly_Salary'].mean())
)

Total_Sum_of_Loan = st.sidebar.slider(
    "Total Sum of Loan",
    int(X['Total_Sum_of_Loan'].min()), int(X['Total_Sum_of_Loan'].max()),
    value=np.int(X['Total_Sum_of_Loan'].mean())
)

Total_Income_for_Join_Application = st.sidebar.slider(
    "Total Income for Join Application",
    int(X['Total_Income_for_Join_Application'].min()), int(X['Total_Income_for_Join_Application'].max()),
    value=np.int(X['Total_Income_for_Join_Application'].mean())
)

Score = st.sidebar.slider(
    "Score",
    int(X['Score'].min()), int(X['Score'].max()),
    value=np.int(X['Score'].mean())
)

# Results
objResult = nb.predict([[
    Credit_Card_Exceed_Months,
    Loan_Amount,
    Loan_Tenure_Year,
    Credit_Card_More_Than_Months,
    Number_of_Dependents,
    Years_to_Financial_Freedom,
    Number_of_Credit_Card_Facility,
    Number_of_Properties,
    Number_of_Bank_Products,
    Number_of_Loan_to_Approve,
    Years_for_Property_to_Completion,
    Number_of_Side_Income,
    Monthly_Salary,
    Total_Sum_of_Loan,
    Total_Income_for_Join_Application,
    Score
]])

st.subheader('Naive Bayes')
st.write(objResult)

st.subheader('Naive Bayes Accuracy')
st.write(nb.score(X_test, y_test))

clf = DecisionTreeClassifier()
clf.get_params()

X_trainCLF, X_testCLF, y_trainCLF, y_testCLF = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

clf = clf.fit(X_trainCLF, y_trainCLF)

y_predClf = clf.predict(X_testCLF)

fn=X.columns

if(obj == 'Employment_Type'):
    cn=list(dict.fromkeys(bank['Employment_Type']))
elif(obj == 'More_Than_One_Products'):
    cn=list(dict.fromkeys(bank['More_Than_One_Products']))
elif(obj == 'Property_Type'):
    cn=list(dict.fromkeys(bank['Property_Type']))
elif(obj == 'State'):
    cn=list(dict.fromkeys(bank['State']))
elif(obj == 'Decision'):
    cn=list(dict.fromkeys(bank['Decision']))

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=300)
tree.plot_tree(clf,
               feature_names = fn, 
               class_names=cn,
               filled = True);

clfImg = io.BytesIO()
fig.savefig(clfImg, format='png')
clfImg.seek(0)
im = Image.open(clfImg)
#im = Image.open("clf.jpeg")
st.subheader('Decision Tree')
st.image(im, width=500, caption="CLF")
clfImg.close()

# im = Image.frombytes('RGB', 
# fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
# st.image(im, width=500, caption="CLF")

st.subheader('Decision Tree Accuracy')
st.write(clf.score(X_trainCLF, y_trainCLF))


X2 = bank.drop([obj], axis=1)
X2 = pd.get_dummies(X2, drop_first=True)
#X2.columns

#ax = sns.relplot(x="Total_Sum_of_Loan", y="Loan_Amount", hue="Employment_Type", data=bank)

km = KMeans(n_clusters = 3, random_state=1)

distortions = []
for i in range(2, 11):
    km = KMeans(
        n_clusters=i, init='random',
        n_init=100, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    distortions.append(km.inertia_)
    
# plot
#plt.plot(range(2,11), distortions, marker='o')
#plt.xlabel('Number of clusters')
#plt.ylabel('Distortion')
#plt.show()

bank_new = bank.copy()
bank_new = bank_new.drop(obj, axis=1)
bank_new[obj]=km.labels_

#fig, axes = plt.subplots(1, 2, figsize=(13,6))     
#sns.scatterplot(x="Loan_Tenure_Year", y="Loan_Amount", hue=obj, data=bank, ax=axes[0])
#sns.scatterplot(x="Loan_Tenure_Year", y="Loan_Amount", hue=obj, data=bank_new, ax=axes[1])

point1 = alt.Chart(bank).mark_point().encode(
    x='Total_Sum_of_Loan',
    y='Loan_Amount',
    color=obj
)

point2 = alt.Chart(bank_new).mark_point().encode(
    x='Total_Sum_of_Loan',
    y='Loan_Amount',
    color=obj
)

st.subheader('K-means cluster')
st.altair_chart(point1)
st.altair_chart(point2)