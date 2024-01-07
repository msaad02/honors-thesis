# Text Retrieval

NOTE TO SELF: Finish pure semantic search class.


---


This section contains the code for the text retrieval portion of the project. 

This section is made completely detached from the data collection step. It sources all the data files from huggingface and processes them for use. To set it up, run the following setup files in order (from the 'text_search' directory). I highly recommend you have a GPU for this step, since it'll most likely take a while otherwise.

1. `./1_train_main_classifier.py`: Contains the code for training the main category classifier.
2. `./2_train_subcat_classifier.py`: Contains the code for training the subcategory classifier.
3. `./3_setup_embeddings.py`: Creates and stores the embeddings for the text data.

And outside the setup, there are also two classes to make the code more useable:

1. `./question_classifier.py`: Contains a class to load the trained models that predict the category and subcategory of a given question.
2. `./text_retriever.py`: Contains a class to oversee the entire text retrieval process. Use this.

You will notice that the retrieval process is not super traditional. We use a combination of the category and subcategory classifiers to narrow down the search space, and then use semantic search to further narrow it down before the reranker ranks the results.

The following is about how it all works, and what decisions were made.

## Training the classifiers

The classifiers we used are trained using the URL data from the scraping process. I.e. text comes from a webpage, and elsewhere in this project we used the text to create questions. So this step aims to predict the category and subcategory (or URL, if you will) of a given question. It's all fairly barebones. We train a small neural network on the data using pytorch. The model itself is ~100k parameters, and the training data varies but is ~24k samples for the main categorizer and for the subcategories it varies from fairly large to rather small. The subcategory predictor is not used for every single possible subcategory. We only train a model if the main category has at least 5 different subcategories, but this could be easily changed. 

To better understand the context of category/subcategory, here are some examples:

- Given a URL: "https://www2.brockport.edu/about", The main category is "about" and the subcategory is "NULL".
- Given a URL: "https://www2.brockport.edu/about/diversity/", the main category is "about" and the subcategory is "diversity".
- Given a URL: "https://www2.brockport.edu/academics/mathematics/", the main category is "academics" and the subcategory is "mathematics".

And so on...

For more technical information about the classifiers, see the docstrings in their training scripts and comments throughout. Evaluation of the models is stored in their respective model directory once they've been trained.

## Using the classifiers

Once we have the classifier it's not super simple to decide their use. There is a lot of grey area in here. For instance, admissions information can be found in the "admissions" section as well as the "graduate" section -- so which one do we use? To address this issue we seek to interpret the logits of the model. The logits are the output of the model before the softmax activation function is applied. They are essentially the raw output of the model. We can interpret these logits as probabilities, and use them to decide which category/subcategory to use. For instance, if the model outputs a probability of 0.8 for the "admissions" category and 0.2 for the "graduate" category, we can say that the model is 80% sure that the question is about admissions. We can then use this information to decide which category/subcategory to use.

In this project, we use the following logic:

1. If the highest probability in the main categories is less than 0.5, indicating low confidence in the prediction:
   - Use all main categories for the search space.
   - Use all subcategories available, if any.
2. If the highest main category probability is 0.5 or higher, suggesting higher confidence in the prediction:
   - Select only those main categories that are within a 0.2 probability range from the highest probability category.
   - Find all subcategories that are within a 0.2 probability range from the highest probability subcategory. (NOTE: subcategories are only selected from the highest probability main category - not a second or third)

As you can see, the classifier is not designed to be restrictive in its predictions. It is designed to be more inclusive than not and rely on the semantic search to narrow down the search space. However, in cases when the classifier is very confident in its prediction, it will be more restrictive. In our testing this is most useful in cases about highly speciifc topics, such as majors, minors, policies, etc. Semantic search is easily confused by mixing broad information with specific questions, so the classifier can be extremely useful in these cases.

All logic to parse responses and probabilites are contained in the `text_retriever.py` file. The `question_classifier.py` file is used to load the models and predict the probabilities.

## Creating the embeddings

The embeddings are an important part of the puzzle. After the categorization results are in, semantic search does the bulk of narrowing down the search space. Creation of embeddings is found in `3_setup_embeddings.py`. Things are pretty simple in this step. There are two main parts to the embedding creation:

1. Chunking and cleaning the data
2. Creating the embeddings using some model of choice

All steps are stored in the setup file, and the output is a pickled dictionary that contains both the raw text files and the embeddings to those text files. Once it's created, you should be ready to go for the retrieval process.

## Retrieval process

The retrieval process is fairly simple. It's all contained in the `text_retriever.py` file. The retrieval process is as follows:

1. Load everything into memory
2. Use the classifier to narrow down the search space
3. Use semantic search to narrow down the search space further
4. Use the reranker to rank the results

Just use the `retrieve` function in the `TextRetriever` class to retrieve results. It takes in a question and returns results. There are some options to change the behavior of the retrieval process, but they are all fairly self-explanatory. There is also an option to return "metadata" about the results, which is just the raw text of the results and the probabilities of the classifier. This is useful for debugging and understanding the behavior of the classifier and semantic search.

# Concluding thoughts

This works pretty well, and it is very quick. There is definitely room for improvement, but it is a marketed improvement over past iterations.

I find that typos in the data, or other abbreviations *significantly* negatively impacts results. The classifier is not very good at handling them, neither is the classifier. Just don't do it! Would not describe this as production ready. 
