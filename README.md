# adobe_best_photo

This project is based on the princeton adobe-photo triage project. (http://phototriage.cs.princeton.edu/). 
The goal is to find the best image from a set of similar images, as a lot of times people take multiple photos of the same scenery and then it's a pain to find the best image of them all. 

Our proposed model involves a siamese neural network architecture which uses a ranking loss function to rank two similar images based on the aesthetic quality. We experimented with two architecture for each siamese network, VGG19 and Inception v3. The model achieved an accuracy of 65 % on the test data. 

