# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:03:33 2022

@author: molan
"""
import os
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import optim
import torch.nn as nn
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from Brain_MRI import Brain_MRI
from Network import VGG_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 5
num_classes = 2
learning_rate = 1.5
batch_size = 128
input_size = 28*28 #Nombre de pixels de l'image = 784
hidden_layers =100
hidden_layers2=100


train_dataset = Brain_MRI(annotations_file= "C:/Users/molan/Documents/Semestre 8/Projet/archive/Brain_Data_Organised/brain_mri_dataset_train_resized.csv", img_dir ="./Train_DataSet_Resized", transform = None)
test_dataset = Brain_MRI(annotations_file="C:/Users/molan/Documents/Semestre 8/Projet/archive/Brain_Data_Organised/brain_mri_dataset_test.csv", img_dir ="./Test_Dataset_Resized", transform =None )

BATCH_SIZE = 32

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""
checkdata = iter(test_loader)   
img, labels = next(checkdata)


for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(img[i][0], cmap='gray')
    plt.title(labels[i].item())
plt.show()
"""

model = VGG_net(in_channels=1, num_classes=2)

#model = Momo(input_size, num_classes) # On crée le modèle avec nb de pixels = 784 et la classification
criterion = nn.CrossEntropyLoss()   # Calcule la fonction de coût classification = CrossENtropyLoss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.5)   # Permet de mettre à jour le poids automatiquement


samples = len(train_loader) # taille de notre jeu d'entraînement
print("Samples - taille du jeu d'entraînement : ",samples)

model = model.to(device) # choix de quel processeur

print("\n---------- Entraînement du modèle ----------\n")

Loss = []
Acc = []

# On entraîne le modèle

for epoch in range(num_epochs) : # Prélèvement d'échantillons de num_epochs (15) de fois 
    model.train()
    for step, (images, labels) in enumerate(train_loader) : # step = numéro de l'énumeration #image = l'image # labels = le chiffre associé 
        images = images.float()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad() # pour éviter l'addition des gradients
        #print(images.shape)
        
        outputs = model(images) #Calcule la sortie de chaque classe (on regardera le max = numéro associé)
        #print(outputs)
        
        loss = criterion(outputs, labels) # calcule la fonction coût
        loss.backward() # rétropropagation
        optimizer.step() # mis à jour des param
        
         
    
    with torch.no_grad() :
        n_correct = 0 # nombre de prédictions correctes -> accuracy
        n_samples = 0 # nombre total d'images dans le jeu de test -> accuracy
        
        for images, labels in test_loader :
            images = images.float()
            images= images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # value, index
            _,predictions_test= torch.max(outputs, 1) # récupère la classe dont la valeur est la plus grande
            n_samples += labels.shape[0]
            n_correct += (predictions_test==labels).sum().item()
            
        acc = 100.0*n_correct/n_samples
        print(f'epoch {epoch}, loss : {loss}, accuracy : {acc}')
        Loss.append(loss.detach().cpu().numpy()) # transformation de tenseur à un array de numpy sinon ça ne fonctionne pas
        Acc.append(acc)

# Afficher les courbes d'erreur et de précision

plt.subplot(1,2,1)
plt.plot(Loss)
plt.title("Fonction Loss")
plt.subplot(1,2,2)
plt.plot(Acc)
plt.title("Accuracy")
plt.show()


print(f'total accuracy for test data : {acc}')  


# Récupérer les poids


print("\n---------- Exemples ----------\n")


for i in range(10):
    step, (images, labels) = next(enumerate(test_loader)) # step = indice d'énumération
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images) # on calcule la sortie des 10 images
    
    _, pred = torch.max(outputs, 1) # on récupère le max pour identifier la classe
    print("\nVérification dans la console : ", pred[i].cpu(), labels[i].cpu())
    
    plt.subplot(5,2,i+1)
    plt.imshow(images[i].cpu().view(28,28), cmap="gray")
    plt.title("Predicted class {}".format(pred[i].cpu()))
    
plt.show()

print("Hello world")
