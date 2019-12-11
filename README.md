**JigsawUnintendedBias**
The Conversation AI team, a research initiative founded by Jigsaw and Google (both part of Alphabet), builds technology to protect voices in conversation. Challenge here is to build machine learning models that detect toxicity and reduce unwanted bias. For example, if a certain minority name is frequently associated with toxic comments, some models might associate the presence of the minority name in a message that is not toxic and wrongly classify the comment as toxic.

**Business Constraints & Understanding as a machine learning problem**
  * Latency requirement is not mentioned. so we consider evaluation time to be a few seconds.
  * Some form of interpretibility of the results.
  * Model Evaluation metric - overall AUC + generalized mean of bias AUC's.

                                                  ---------------
# Data Overview
* training data has 1289508 rows
* validation data has 161193 rows
* test data has 161233 rows
* useful training data columns are: 
  * 'id' - **Comment ID**
  * 'comment_text' - **Text to evaluate**
  * 'toxicity' - **Target Variable**

  * 'severe_toxicity', 'obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat', 'male', 'female', 'transgender',        'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation', 'christian',               'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion', 'black', 'white', 'asian',                                'psychiatric_or_mental_illness',  'identity_annotator_count', 'toxicity_annotator_count'  -  **Identity Variables**


  * 'male', 'female', 'homosexual_gay_or_lesbian','christian', 'jewish', 'muslim', 'black', 'white', 'asian' - **Identity variables to validate in the kaggel competition**
  
# Observations from EDA : 
  * We found only 8% of data is toxic. Data is imbalanced.
  * Identities [FEMALE, MALE, BLACK, WHITE, CHRISTIAN] found mostly in the Training data comments. toxicity percentage is high in comments with [homosexual_gay_or_lesbian] .
  * New features[Number of words in a comment, Number of expressive characters(?!. etc.) in a comment] are extracted. 
  num_words in the range[100, 180]approx. and num_expr_words in the range[100, 150]approx. in a comment tends to be a non-toxic comment.
  
# Text Preprocessing
Preprocessing techniques followed :
  * removing special characters and punctuation marks except [?!].
  * replacing markup text, ‘https://’, spaces, numerics, with an empty string.
  * expanding language contractions (eg. don’t -> do not)
  * removing stopwords using nltk lib except ‘not’ keyword.
  * word lemmatization using nltk lib.
  
Note : Considered only text data as a clean resource for the problem statement. Other identities have been used wile training the model to reduce the loss and improve model performance.

# Comment word Tokenization and Embedding matrix
Tokenizer assigns a rank or a token to each word in the comments based on the frequency of the word in all comments. Fit the tokenizer with train and test data comments to give our model maximum vocabulary as we are not using pre-trained models here. Use pad_sequences() to ensure that all comments have the same length and to train the model in batches. It's good to have the maximum sequence for padding between 200–250 based on num-words distribution from the above plot. I’ve chosen 220.

Embedding layer, using the embedding matrix, will generate a continuous vector representation for each word that is represented as a token by the tokenizer. I have used Glove, Crawl 300d word vectors, and concatenated two word-vectors to get maximum context for a word.

# Model Architecture 
Two Bidirectional LSTM’s give the context of a sequence for NLP use cases most of the time. 1Dimensional Dropout at the start and MaxPooling with valid padding gave good results. Using sample weights while training the model is a trick to avoid false positives for the model.

* sd = SpatialDropout1D(DROPOUT_RATE)(embedding_layer)
* x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(sd)
* x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
* x = keras.layers.MaxPooling1D(2, padding='valid')(x)
* x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
* x = Dropout(DROPOUT_RATE)(x)
* x = Flatten()(x)
* x = Dense(128, activation='relu')(x)
* result = Dense(1, activation='sigmoid')(x)
* aux_result = Dense(aux_target_count, activation='sigmoid')(x)

Note : While Training the model, we use sample weights to the training data. We increase the weight of 1/4 for each comment for each case a comment having identity feature/s, toxic & not having an identity feature, non-toxic & having an identity feature.

**Parameters:**
Total params: 169,631,630 
Trainable params: 5,143,430 
Non-trainable params: 164,488,200

**Output**
AUC score - 0.920 - as the final metric .
bpsn_auc[background positive and subgroup negative] is is low for black, white, homosexual_gay_or_lesbian. This states that a maximum of 15% of the comments with identity[black] are classified as positive[toxic] even if they are negative[non-toxic], etc.
# Conclusions:
  * Used GLOVE and FAST CRAWL 300d embeddings for comment text vectorization and modified sample weights as an improvement.     * Defined bias metrics from google benchmark kernel.
  * A major improvement can be achieved with context-based word-embeddings using bert, xlnet models and finetuning them.


### Model Predicted with a combined AUC score of 0.924.
