import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# reading the data csv file into a dataframe
df = pd.read_csv('KNN_Project_Data')
print(df.head())

# using seaborn to create a pairplot with hue indicated by the TARGET CLASS column
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')
plt.show()

# standardizing the variables

# creating a scaler object
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))

# converting scaled features to a dataframe
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())

# splitting data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)

# creating a KNN model instance with k=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

# predictions and evaluations

prediction = knn.predict(X_test)
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

# choosing a k value

# loop that trains various KNN models with different k values
# and keep track of the error_rate for each models
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# plotting the result
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

# retain model with new k value
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
