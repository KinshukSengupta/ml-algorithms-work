Emotion Detection using CNN

Data cleaning
The data cleaning code implemented in Python does the following things:
1.	Replace short words like don’t to do-not. Can’t to can-not
2.	Stopword removal from the nltk toolkit
3.	Removes Hyperlinks with the keyword LINK
4.	Removes mentions and replaces with the keyword MENTION

Word embedding
1.	Different combinations of the glove embedding were tried. The results are given in this table.
a.	Average of word glove vectors
b.	Weighted average of glove vectors by tf-idf score
c.	Stacking n-dimension glove vector next to each other
d.	In the CNN implementation. The tweets are represented by a 30x200 dimension matrix. 
2.	The words which could not be found in the glove vector corpus were given small random values using the function np.random.randn(1,EMBEDDING_DIM) /100
Feature dimension 	100 x 30	200 x 30	300 x 30
Neural Network	51.75%	56.38%	58.95%
Deep Belief Network	46.61%	54.92%	57.49%

Using the 300 dimension glove vector gives the best accuracy
 
	Figure 2 :  Confusion matrix for DBN
Using a CNN where each document is a 2-d vector represented by a 30 x 200. (30 is the padded length, 200 is the dimension of the glove vector)
Specifications of the CNN
Number of filters	3
Filter sizes	(3,200) , (4,200) , (5,200)
Max Pooling	(28,1) , (27,1) , (26,1)
num_filters	512
Dropout regularization	.3
Epochs	30
Accuracy	86.6%
Optimizer	Adam
Learning rate	.001
Beta_1	.9
Beta_2	.999

 
Figure 3 : Confusion Matrix for CNN
Hyper-parameter tuning
The hyper-parameters were tuned orthogonally. A single hyper-parameter was searched over a random space. And the best one was chosen based on performance on the test set.
 
Figure 4 : Validation curve for the learning rate

