# import pandas as pd



# # 2) MULTIPLE LINEAR REGRESSION :



# dataset = pd.read_csv("placement.csv")

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.pairplot(data=dataset)
# plt.show()

# x = dataset.iloc[:,:-1]
# y = dataset["cgpa"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# from sklearn.linear_model import LinearRegression

# var = LinearRegression()

# var.fit(x_train,y_train)

# print(var.score(x_train,y_train)*100)

# print(var.coef_)

# print(var.predict(x_test))








# POLYNOMIAL REGRESSION :




# import matplotlib.pyplot as plt
# dataset = pd.read_csv("polynomial.csv")

# plt.scatter(dataset["Level"],dataset["Salary"])
# plt.xlabel("Level")
# plt.ylabel("Salary")
# plt.show()

# x = dataset[["Level"]]
# y = dataset["Salary"]

# from sklearn.preprocessing import PolynomialFeatures

# pf = PolynomialFeatures(degree=2)
# pf.fit(x)
# print(pf.transform(x))

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)

# from sklearn.linear_model import LinearRegression

# var = LinearRegression()

# var.fit(x_train,y_train)
# print(var.score(x_test,y_test)*100)

# prd = var.predict(x)

# plt.scatter(dataset["Level"],dataset["Salary"])
# plt.plot(dataset["Level"],prd,c="red")
# plt.xlabel("Level")
# plt.ylabel("Salary")
# plt.legend(["org","prd"])
# plt.show()

# print( var.coef_)

# print( var.intercept_)

# print(var.predict(x_test))












# L1,L2 (PRACTICE):




# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split

# dataset = pd.read_csv("placement.csv")

# # sns.heatmap(data=dataset.corr(),annot=True)
# # plt.show()

# x = dataset.iloc[:,:-1]
# y = dataset["cgpa"]

# sc = StandardScaler()
# sc.fit(x)
# print(sc.transform(x))

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# from sklearn.linear_model import LinearRegression ,Lasso , Ridge

# var = LinearRegression()
# var.fit(x_train,y_train)

# print(var.score(x_test,y_test)*100)

# plt.bar(x.columns,var.coef_)
# plt.show()


# from sklearn .metrics import mean_absolute_error,mean_squared_error
# import numpy as pd

# print(mean_squared_error(y_test,var.predict(x_test)))
# print(mean_absolute_error(y_test,var.predict(x_test)))






# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)

# sk = Lasso(alpha=0.3)
# sk.fit(x_train,y_train)
# print(sk.score(x_test,y_test)*100)

# plt.bar(x.columns,sk.coef_)
# plt.title("Lasso")
# plt.xlabel("columns")
# plt.ylabel("coef")
# plt.show()

# from sklearn .metrics import mean_absolute_error,mean_squared_error
# import numpy as pd

# print(mean_squared_error(y_test,sk.predict(x_test)))
# print(mean_absolute_error(y_test,sk.predict(x_test)))






# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)

# ss = Ridge(alpha=10)
# ss.fit(x_train,y_train)
# print(ss.score(x_test,y_test)*100)

# plt.bar(x.columns,ss.coef_)
# plt.title("Ridge")
# plt.xlabel("columns")
# plt.ylabel("coef")
# plt.show()

# from sklearn .metrics import mean_absolute_error,mean_squared_error
# import numpy as pd

# print(mean_squared_error(y_test,ss.predict(x_test)))
# print(mean_absolute_error(y_test,ss.predict(x_test)))











# PRACTICE ON LOGISTIC REGRESSION:





# import seaborn as sns
# import matplotlib.pyplot as plt
# dataset = pd.read_csv("network.csv")
# from sklearn.model_selection import train_test_split


# x = dataset[["Age"]]
# y = dataset["Purchased"]

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)

# from sklearn.linear_model import LogisticRegression

# var = LogisticRegression()
# var.fit(x_train,y_train)
# print(var.predict([[30]]))

# print(var.score(x_test,y_test)*100)

# sns.scatterplot(x="Age",y="Purchased",data=dataset)
# sns.lineplot(x = "Age",y = var.predict(x),data=dataset,color = "red")
# plt.show()





# sns.scatterplot(x="Age",y="Purchased",data=dataset)
# # plt.show()

# x = dataset[["Age"]]
# y = dataset["Purchased"]





# PRACTICE ON LOGISTIC REGRESSION (MULTIPLE INPUT):
# HERE DATA IS NOT GOOD 



# import seaborn as sns
# import matplotlib.pyplot as plt

# dataset = pd.read_csv("placement.csv")
# print(dataset.head(5))

# sns.scatterplot(x="cgpa",y="package",data=dataset,hue="placed")
# plt.show() 


# x = dataset.iloc[:,:-1]
# y = dataset["placed"]


# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=44)

# from sklearn.linear_model import LogisticRegression

# var = LogisticRegression()
# var.fit(x_train,y_train)

# print(var.score(x_test,y_test)*100)

# print(var.predict([[6.89,3.26]]))

# print(var.coef_)

# print(var.intercept_)

# from mlxtend.plotting import plot_decision_regions

# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=var)
# plt.show()











# PRACTICE ON LOGISTIC REGRESSION (POLYNOMIAL INPUT):




# import seaborn as sns
# import matplotlib.pyplot as plt

# dataset = pd.read_csv("placement.csv")


# sns.scatterplot(x="cgpa",y="package",data=dataset,hue="placed")
# # plt.show() 

# x = dataset.iloc[:,:-1]
# y = dataset["placed"]


# from sklearn .preprocessing import PolynomialFeatures

# pf = PolynomialFeatures(degree=4)
# x_poly = pf.fit_transform(x)

# feature_names = pf.get_feature_names_out(input_features=x.columns)
# x = pd.DataFrame(x_poly,columns=feature_names)
# print(x)



# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=46)

# from sklearn.linear_model import LogisticRegression

# var = LogisticRegression()
# var.fit(x_train,y_train)

# print(var.score(x_test,y_test)*100)











# PRACTICE ON LOGISTIC REGRESSION (MULTICLASS CLASSIFICATION):




# import seaborn as sns
# import matplotlib.pyplot as plt

# dataset = pd.read_csv("placement.csv")



# print(dataset["placed"].unique())

# sns.pairplot(data=dataset)
# plt.show()

# x = dataset.iloc[:,:-1]
# y = dataset["placed"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=46)

# from sklearn.linear_model import LogisticRegression

# sk = LogisticRegression(multi_class="multinomial")
# sk.fit(x_train,y_train)
# print(sk.score(x_test,y_test)*100)









# CONFUSION MATRRIX:




# import seaborn as sns
# import matplotlib.pyplot as plt

# dataset = pd.read_csv("placement.csv")

# x = dataset.iloc[:,:-1]
# y = dataset["placed"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=46)

# from sklearn.linear_model import LogisticRegression

# ss = LogisticRegression()
# ss.fit(x_test,y_test)
# print(ss.score(x_test,y_test)*100)


# from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

# cf = confusion_matrix(y_test,ss.predict(x_test))
# print(cf)


# sns.heatmap(cf,annot=True)
# plt.show()

# print(precision_score(y_test,ss.predict(x_test))*100)

# print(recall_score(y_test,ss.predict(x_test))*100)

# print(f1_score(y_test,ss.predict(x_test))*100)











# IMBALANCED DATASET:





# dataset = pd.read_csv("placement.csv")

# print(dataset["placed"].value_counts())

# x = dataset.iloc[:,:-1]
# print(x)
# y = dataset["placed"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=46)

# from sklearn.linear_model import LogisticRegression

# sk = LogisticRegression()
# sk.fit(x_train,y_train)

# print(sk.score(x_test,y_test)*100)

# print(sk.predict([[6.89,3.26]]))






# BALANCING ABOVE DATA:(USING UNDER SAMPLING)




# dataset = pd.read_csv("placement.csv")

# x = dataset.iloc[:,:-1]
# y = dataset["placed"]

# print(dataset["placed"].value_counts())

# from imblearn.under_sampling import RandomUnderSampler

# ru = RandomUnderSampler()

# ru_x, ru_y = ru.fit_resample(x,y)

# print(ru_y.value_counts())


# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=46)

# from sklearn.linear_model import LogisticRegression

# sk = LogisticRegression()
# sk.fit(x_train,y_train)

# print(sk.score(x_test,y_test)*100)

# print(sk.predict([[7.89,2.99]]))












# PRACTICE ON NAIVE BAYES:(DATA SHOULD BE NORMAL DISTRIBUTION)



# import seaborn as sns
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions

# dataset = pd.read_csv("placement.csv")

# sns.scatterplot(x="cgpa",y="package",data=dataset,hue="placed")
# plt.show()

# x = dataset.iloc[:,:-1]
# y = dataset["placed"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=46)

# from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

# gnb = GaussianNB()
# gnb.fit(x_train,y_train)

# print(gnb.score(x_test,y_test)*100,gnb.score(x_train,y_train)*100)

# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=gnb)
# plt.show()



# mnb = MultinomialNB()
# mnb.fit(x_train,y_train)

# print(mnb.score(x_test,y_test)*100,mnb.score(x_train,y_train)*100)

# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=mnb)
# plt.show()



# bnb = BernoulliNB()
# bnb.fit(x_train,y_train)

# print(bnb.score(x_test,y_test)*100,bnb.score(x_train,y_train)*100)

# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=bnb)
# plt.show()



# END






























# DEEP LEARNING :





# PERCEPTRON WORK:




# import seaborn as sns
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_decision_regions

# dataset = pd.read_csv("placement.csv")

# sns.scatterplot(x="cgpa",y="package",data=dataset,hue="placed")
# plt.show()

# x = dataset.iloc[:,:-1]
# y = dataset["placed"]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=11)


# from sklearn.linear_model import Perceptron

# pr = Perceptron()
# pr.fit(x_train,y_train)

# print(pr.score(x_test,y_test)*100)


# plot_decision_regions(x.to_numpy(),y.to_numpy(),clf=pr)
# plt.show()











# MULTI-LAYER PERCEPTRON (ANN):


# REMOVING OVERFITTING,IDENTIFYING OVERFITTING IN MODEL + ETC THINGS:





# dataset = pd.read_csv("placement.csv")


# input_data = dataset.iloc[:,:-1]
# output_data = dataset.iloc[:,-1]

# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test = train_test_split(input_data,output_data,test_size=0.2,random_state=46)


# sk = StandardScaler()
# print(sk.fit_transform(input_data))


# import tensorflow
# from keras.callbacks import EarlyStopping
# from keras.layers import Dense
# from keras.models import Sequential

# ann = Sequential()

# ann.add(Dense(1,input_dim = 2,activation="relu"))

# ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

# print(ann.fit(x_train,y_train,batch_size=10,epochs=50,validation_data=(x_test,y_test),callbacks = EarlyStopping()))




# from sklearn.metrics import accuracy_score

# prd = ann.predict(x_test)

# prd_data = []
# for i in prd : 
#     if i[0]>0.5:
#         prd_data.append(1)
#     else:
#         prd_data.append(0)    

# print(accuracy_score(y_test,prd_data)*100)



# prd1 = ann.predict(x_train)

# prd_data1 = []
# for i in prd1 : 
#     if i[0]>0.5:
#         prd_data1.append(1)
#     else:
#         prd_data1.append(0)    

# print(accuracy_score(y_train,prd_data1)*100)

# train_accuracy = ann.history.history["accuracy"]

# test_accuracy = ann.history.history["val_accuracy"]

# plt.plot(range(1,len(train_accuracy)+1),train_accuracy)

# plt.plot(range(1,len(test_accuracy)+1),test_accuracy,c="red")
# plt.show()












# CNN PRACTICE:(BELOW PROGRAM HAS QUIT ERROR)





# import tensorflow as tf

# from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator


# cnn = Sequential()

# cnn.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation="relu"))
# cnn.add(MaxPooling2D(pool_size=(2,2)))
# cnn.add(Conv2D(16,(3,3),activation="relu"))
# cnn.add(MaxPooling2D(pool_size=(2,2)))
# cnn.add(Flatten())

# cnn.add(Dense(64,activation="relu"))
# cnn.add(Dense(32,activation="relu"))
# cnn.add(Dense(16,activation="relu"))
# cnn.add(Dense(8,activation="relu"))
# cnn.add(Dense(1,activation="sigmoid"))

# cnn.compile(loss="binary_crossentropy",optimizer="adam")


# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         "C:\data set\cnn\train_data",
#         target_size=(64,64),
#         batch_size=32,
#         class_mode='binary')

# test_generator = test_datagen.flow_from_directory(
#         'C:\data set\cnn\test_data',
#         target_size=(64,64),
#         batch_size=32,
#         class_mode='binary')

# cnn.fit_generator(
#         train_generator,
#         steps_per_epoch=20,
#         epoch=5,
#         validation_data=test_generator)
#NEW TEST








# END





