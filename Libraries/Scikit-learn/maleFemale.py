from sklearn import tree

#[height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
#here X be the List of lists of [Height,weight,shoe size] of 11 peoples.
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
#here Y is dataset of gender of above 11 peoples 
	
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190, 70, 43]])

print(prediction)

#https://medium.com/@ry007/classification-using-decision-tree-8437ffcc2c89
