import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px

# plt.style.use('default')
color_pallete = ['#fc5185', '#3fc1c9', '#364f6b']
sns.set_palette(color_pallete)
sns.set_style("white")



df = pd.read_csv("Iris.csv")
X = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values

#visualize
fig = px.scatter_3d(df, x="PetalLengthCm", y="PetalWidthCm", z="SepalLengthCm", size="SepalWidthCm", 
              color="Species", color_discrete_map = {"Joly": "blue", "Bergeron": "violet", "Coderre":"pink"})
fig.show()


#model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=43)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#optimal value of k calculated later
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

#single prediciton
#ip = [5.0,3.3,1.4,0.2] #set
#ip = [7.0,3.2,4.7,1.4] #ver
ip = [6.4,2.7,5.3,1.9] #vir
print('Prediction: ',knn.predict([ip]))


pred = knn.predict(x_test)
acc = accuracy_score(y_test,pred)
print('Accuracy: ',round(acc*100,2))



#cross validation to find best k
from sklearn.model_selection import cross_val_score
k_list = list(range(1,50,2))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())


MSE = [1 - x for x in cv_scores]
plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)
plt.show()

#use graph to determine optimal k
