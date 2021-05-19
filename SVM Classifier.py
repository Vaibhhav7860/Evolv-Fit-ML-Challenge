# Level 1 :

# Creating a SVM Classifier, to find out the name of the cricketer from the given image.



# Importing Necessary Modules

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# This code is commented because I have stored the complete data images of all the cricketers in a pickle file named
# data1.pickle, once the complete data is stored in the pickle file there is no need of this code.
'''
# Setting the directory of our dataset
directory = "dataset1/images"

Player_labels = ["bhuvneshwar_kumar", 'dinesh_karthik', 'hardik_pandya','jasprit_bumrah','k._l._rahul','kedar_jadhav','kuldeep_yadav','mohammed_shami','ms_dhoni','ravindra_jadeja','rohit_sharma','shikhar_dhawan','vijay_shankar','virat_kohli','yuzvendra_chahal']

data = []

for player in Player_labels:
    path = os.path.join(directory, player)
    label = Player_labels.index(player)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        player_img = cv2.imread(imgpath, 0)
        try:
            player_img = cv2.resize(player_img,(100,100))
            image = np.array(player_img).flatten()
            data.append([image, label])
        except Exception as e:
            pass
#print(len(data))

pick_in = open('data1.pickle','wb')
pickle.dump(data, pick_in)
pick_in.close()
'''

# Opening and loading the data1.pickle file
pick_in = open('data1.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()


# Randomly shuffling our data so that at the time of testing each time we get a different test image.
random.shuffle(data)
features = []
labels = []

for feature,label in data:
    features.append(feature)
    labels.append(label)


# Splitting our data into training and testing data.
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2)

# This code is commented after training our SVM Model.
'''
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(X_train, Y_train)
'''


# Saving the output file of our model.
pick = open('model.save', 'rb')
model = pickle.load(pick)
pick.close()

# Making the predictions.
prediction = model.predict(X_test)

# Calculating the Accuracy.
accuracy = model.score(X_test, Y_test)


# I have tried to represent different players to their respective classes.
Player_labels = ["Bhuvneshwar Kumar", 'Dinesh Karthik', 'Hardik Pandya','Jasprit Bumrah','K.L. Rahul','Kedar Jadhav','Kuldeep Yadav','Mohammed Shami','Ms Dhoni','Ravindra Jadeja','Rohit Sharma','Shikhar Dhawan','Vijay Shankar','Virat Kohli','Yuzvendra Chahal']


# Printing the accuracy and our Predicted Player.
print("Accuracy : ", accuracy)

print('Prediction is : ',Player_labels[prediction[0]])

# Plotting the image of the tested player.
Player1 = X_test[0].reshape(100,100)
plt.imshow(Player1,cmap='gray')
plt.show()
