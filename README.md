# Detecting the best photo from a series

This project was a part of a course project for the CS 689, advanced machine learning course at UMASS. It is based on the princeton adobe-photo triage project. (http://phototriage.cs.princeton.edu/). 
The goal is to find the best image from a set of similar images, as a lot of times people take multiple photos of the same scenery and then it's a pain to find the best image of them all. 

Our proposed model involves a siamese neural network architecture which uses a ranking loss function to rank two similar images based on the aesthetic quality. We experimented with two architecture for each siamese network, VGG19 and Inception v3. The model achieved an accuracy of 65 % on the test data. 

Used Pytorch.
The dataset can be obtained here. (http://phototriage.cs.princeton.edu/download.html)
