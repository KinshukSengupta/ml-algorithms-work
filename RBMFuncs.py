def encode_glove_unigram(X_train , X_test , embedding_dim , word_index):
    """
    Encodes glove vectors in a document
    
    Return value:
    - A training set and a test set with glove embeddings
    """
    import os
    import itertools
    import numpy as np
    
   
    #Embedding the vector in this step
    EMBEDDING_DIM = embedding_dim
    FILE_NAME = "glove.6B." + str(embedding_dim) + "d.txt"
    
    #This is very time consuming, do not run again and again
    GLOVE_DIR = '/emdedding/'

    embeddings_index = {} #dictionary keys - words and values are the 100 dimension vector
    embeddings_index = {} #dictionary keys - words and values are the 100 dimension vector
    f = open(os.path.join(GLOVE_DIR, FILE_NAME)  , encoding="utf8")
    for line in f:
        values = line.split() #each line in the glove text file
        word = values[0] #The thing on the 0th position is the word
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    
    notFoundCounter = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            notFoundCounter+=1
            
    print("Vectors not found: " +str(notFoundCounter))
    
    print("The dimention of the embedding Matrix should be : (nx , EMBEDDING_DIM)")
    print("And it actually is: " + str(embedding_matrix.shape))
    
   
    zeroList = np.zeros((1,EMBEDDING_DIM)).tolist()[0]
    x_train_glove = np.zeros((X_train.shape[0] , EMBEDDING_DIM * 30))
    row = 0
    for document in X_train:
        #Looping over each Tweet
        vectorTemp = [] #initializing the vector representation of the tweet
        #This will become a 1 x 300 vector
        for word in document:
            if word == 0:#This is the padded one
                vectorTemp.append(zeroList)
            else:
                vectorTemp.append(embedding_matrix[word , :].tolist())
        vectorTemp = list(itertools.chain.from_iterable(vectorTemp))#The original vectorTemp is a list of lists.
        #And this helps us in decomposing it
        x_train_glove[row] = vectorTemp
        row+=1

    x_test_glove = np.zeros((X_test.shape[0] , EMBEDDING_DIM * 30))
    row = 0
    for document in X_test:
        #Looping over each Tweet
        vectorTemp = [] #initializing the vector representation of the tweet
        #This will become a 1 x 300 vector
        for word in document:
            if word == 0:#This is the padded one
                vectorTemp.append(zeroList)
            else:
                vectorTemp.append(embedding_matrix[word , :].tolist())
        vectorTemp = list(itertools.chain.from_iterable(vectorTemp))#The original vectorTemp is a list of lists.
        #And this helps us in decomposing it
        x_test_glove[row] = vectorTemp
        row+=1

    print("Shape of Training set now is: " + str(x_train_glove.shape))
    print("Shape of Test set now is: " + str(x_test_glove.shape))
    
    return (x_train_glove , x_test_glove)
	
def create_Model(UNITS , DROPOUT , REGU , LEARNING_RATE ,INPUT_DIM):
    
    #Defining the network architecture
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import History
    from keras.layers import Dropout
    from keras import regularizers
  
    units , dropout , regu = UNITS , DROPOUT , REGU
    assert(len(units) == len(dropout))
    layers = list(range(0 , len(units)))
    classifier = Sequential()

    #Addding the layers iteratively
    for u , d , l , r in zip(units , dropout , layers , regu):
        if l == 0: #For the input layer
            classifier.add(Dense(units = u, kernel_initializer = 'uniform' , activation = 'relu', input_dim =INPUT_DIM , 
                                kernel_regularizer=regularizers.l2(r)))
            classifier.add(Dropout(d))
           # print("First layer added...")
        else:
            #Print("Adding another layer...")
            classifier.add(Dense(units = u, kernel_initializer = 'uniform' , activation = 'relu' , 
                                kernel_regularizer=regularizers.l2(r)))
            classifier.add(Dropout(d))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    #print("Architecture defined...")
    #Architecture defined
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    history = History()
    myAdam = keras.optimizers.Adam(lr= LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    classifier.compile(optimizer = myAdam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print("Compiled")
    return classifier
	
	
#Converting y_train as one hot encoded vectors
def convertYToDummy(y):
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    return np_utils.to_categorical(encoded_Y)#keras take an input in this form
	
def accuracyFunc(X_test , y_test , model):
    from sklearn.metrics import accuracy_score
    y_prob = model.predict(X_test) 
    y_pred = y_prob.argmax(axis=-1)
    y_actual = y_test
    score = accuracy_score(y_true = y_actual , y_pred = y_pred )
    return score
	
def clean_text(raw_text):
    import re
    text = raw_text#.encode('b')
    useReplaceDict = {"don't" : "do not", "can't" : "can not", "doesn't" : "does not", "hasn't": "has not", 
            "haven't" : "have not", "hadn't" : "had not", "couldn't" : "could not", "wasn't" : "was not", "didn't": "did not",
             "weren't" : "were not", "wouldn't": "would not", "won't" : "would not", "shouldn't": "should not", 
            "don't": "do not", "it's" : "it is", "he's" : "he is", "she's" : "she is", "i'm" : "i am", "i've" : "i have", 
            "they've": "they have", "you've" : "you have", "we've": "we have", "ppl" : "people", "we're" : "we are", 
            "i'll": "i will" , "they're" : "they are", "he'll" : "he will", "she'll" : "she will", "we'll" : "we will" ,
            "they'll":"they will", "you'll" : "you will", "w/o" : "without out", "i'd" : "i would"}
    from nltk.corpus import stopwords
    nltk_stop_words = set(stopwords.words('english'))
    #text = " ".join([x for x in text.split("\n")]) #Joins new lines with spaces
    text = text.replace("\n"," ").replace("\r"," ").replace("#"," ").lower()
    text = re.sub(r"https\S+", "LINK", text)
    text = text.replace("&amp;", "and")
    text = re.sub(r'/', ' or ', text)
    text = re.sub(r'@\S+', 'MENTION',text)
    text = re.sub(r'[",:.\-;()|+&=!/?#$@\[\]]+', ' ', text)
    #text = re.sub(r'www.+' , '' , text)
    replaced_text = ""
    rep_text = useReplaceDict
    if useReplaceDict:
        for term in text.split():
            if term.strip() != '' and term.strip() in rep_text.keys():
                replaced_text += rep_text[term] + " "
            else:
                replaced_text += term + " "    
        text = replaced_text
    text = " ".join([x for x in text.split() if x.strip() not in nltk_stop_words])
    text = re.sub(r'[",:.\-;()|+&=!?#$@\[\]]+', ' ', text)
#     print replaced_text
    return text#.encode("utf-8")
	
def encode_glove_average(X_train , X_test , embedding_dim  , word_index):
    """
    Encodes glove vectors in a document
    """
    import os
    import numpy as np
    #Embedding the vector in this step
    EMBEDDING_DIM = embedding_dim
    FILE_NAME = "glove.6B." + str(embedding_dim) + "d.txt"
    FILE_NAME = "glove.6B." + str(embedding_dim) + "d.txt"
    
    #This is very time consuming, do not run again and again
    GLOVE_DIR = 'C:/Users/Nipun.Puri/Desktop/wordEmbed/'

    embeddings_index = {} #dictionary keys - words and values are the 100 dimension vector
    f = open(os.path.join(GLOVE_DIR, FILE_NAME)  , encoding="utf8")
    for line in f:
        values = line.split() #each line in the glove text file
        word = values[0] #The thing on the 0th position is the word
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.ranom.rand(len(word_index) + 1, EMBEDDING_DIM) / 100
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be the same randomly intialized, better than zero
            embedding_matrix[i] = embedding_vector
    
    print("The dimention of the embedding Matrix should be : (nx , EMBEDDING_DIM)")
    print("And it actually is: " + str(embedding_matrix.shape))
    
    import itertools

    zeroList = np.zeros((1,EMBEDDING_DIM)).tolist()[0]
    x_train_glove = np.zeros((X_train.shape[0] , EMBEDDING_DIM))
    row = 0
    for document in X_train:
        #Looping over each Tweet
        vectorTemp = np.zeros((1,EMBEDDING_DIM)) #initializing the vector representation of the tweet
        #This will become a 1 x 300 vector
        for word in document:
            if word == 0:#This is the padded one
                vectorTemp = vectorTemp + np.zeros((1,EMBEDDING_DIM))
            else:
                vectorTemp = vectorTemp + np.array(embedding_matrix[word , :])
        vectorTemp = vectorTemp / 30
        #And this helps us in decomposing it
        x_train_glove[row] = vectorTemp
        row+=1

    x_test_glove = np.zeros((X_test.shape[0] , EMBEDDING_DIM))
    row = 0
    for document in X_test:
        #Looping over each Tweet
        vectorTemp = np.zeros((1,EMBEDDING_DIM)) #initializing the vector representation of the tweet
        for word in document:
            if word == 0:#This is the padded one
                vectorTemp = vectorTemp + np.zeros((1,EMBEDDING_DIM))
            else:
                vectorTemp = vectorTemp + np.array(embedding_matrix[word , :])
            vectorTemp = vectorTemp / 30
        #And this helps us in decomposing it
        x_test_glove[row] = vectorTemp
        row+=1

    print("Shape of Training set now is: " + str(x_train_glove.shape))
    print("Shape of Test set now is: " + str(x_test_glove.shape))
    
    return (x_train_glove , x_test_glove)
	
def create_Model_binary(UNITS , DROPOUT , REGU , LEARNING_RATE ,INPUT_DIM):
    
    #Defining the network architecture
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import History
    from keras.layers import Dropout
    from keras import regularizers
  
    units , dropout , regu = UNITS , DROPOUT , REGU
    assert(len(units) == len(dropout))
    layers = list(range(0 , len(units)))
    classifier = Sequential()

    #Addding the layers iteratively
    for u , d , l , r in zip(units , dropout , layers , regu):
        if l == 0: #For the input layer
            classifier.add(Dense(units = u, kernel_initializer = 'uniform' , activation = 'relu', input_dim =INPUT_DIM , 
                                kernel_regularizer=regularizers.l2(r)))
            classifier.add(Dropout(d))
           # print("First layer added...")
        else:
            #Print("Adding another layer...")
            classifier.add(Dense(units = u, kernel_initializer = 'uniform' , activation = 'relu' , 
                                kernel_regularizer=regularizers.l2(r)))
            classifier.add(Dropout(d))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    #print("Architecture defined...")
    #Architecture defined
    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    history = History()
    myAdam = keras.optimizers.Adam(lr= LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    classifier.compile(optimizer = myAdam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.compile(optimizer = myAdam, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print("Compiled")
    return classifier

#Tokenizing the words into integers, and creating the word_index, where the ith row is the vector representation of the word
#represented by the integer i
def embedding_matrix(corpus , embeddings_index , EMBEDDING_DIM):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import numpy as np
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus) #This is the text converted into integers
    word_index = tokenizer.word_index
    #Creating the embedding_matrix
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    #In this matrix, I want the ith row to represent the glove vector of the word represented by the integer i
    #dictionary mapping words and their integer representation
    notFound = 0
    for word, i in word_index.items():
        #i is the integer representation of word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            notFound+=1

    print("The dimention of the embedding Matrix should be : (nx , EMBEDDING_DIM)")
    print("And it actually is: " + str(embedding_matrix.shape))
    print(str(notFound) + " words were not found.")
    
    #Representing the documents as integers
    dataInt = np.zeros((corpus.shape[0] , 30) , dtype = int)
    for i in range(0,len(sequences)):
        dataInt[i][:len(sequences[i])] = sequences[i]
    
    return embedding_matrix , word_index , dataInt
	
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	
	
def removeDuplicates(data):
    import pandas as pd
    import numpy as np
    texts = {}
    for text in data.text:
        if text not in texts:
            texts[text] = 1
        else:
            texts[text]+=1
            
    data['count'] = np.zeros((data.shape[0] , 1))
    for i in range(0 , data.shape[0]):
        textTemp = data.text[i]
        data['count'][i] = texts.get(textTemp)
    data = data.sort_values(['count' , 'text' , 'score'] , ascending = [0,0,0])
    
    for i in range(0,data.shape[0]-3):
        previous = data['text'][data.index[i]]
        current = data['text'][data.index[i+1]]
        if current == previous:
            previous = current
            data.drop(data.index[i] , inplace = True)
        
    return data