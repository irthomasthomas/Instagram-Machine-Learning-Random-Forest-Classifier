This project was a learning exercise. An exploration of machine-learning through the training and deployment of a Random Forest Classifier.

The project scales the walled-garden of Instagram to find posts containing items for sale from instagram citizens, and make them searchable. It was conceived of at a time when ML was still mostly for researchers and big-tech. Tools for continous training and deployment where few. I used a new stack of tools for machine-learning from redis . Being alpha software, it changed frequently, thus, using this project, tody, would be ill-advised without significant re-writing. 

I personally classified thousands of posts, and then used methods to generate further synthetic data. 

The main tools used where
Redis to store and serve the model
RedisGears to run the pre-processing pipeline
Scikit-learn to train the model
pytorch
onnx 
Svelte JS

To make serving efficient from my laptop, I made liberal use of probabilistic data-structures such bloom-filter, hyperloglog, and count-min-sketch.
 The probabilistic data-structures allowed me to run a fully-functional public demo on a single small VM in a deterministic manner.

Briefly, the program consists of:
  - Scraping instagram posts from hashtags related selling.  
  - Manually tagging posts as relevant or not.
  
  - Passing the post text through a pre-processing pipeline in python:
  * Cleaning
  * Tokenizing
  * Lemmatizing
  * Removing stop-words
   
  - Training a Random-Forest-Classifier on the cleaned text.
  - Converting the RFC to matrix
  - Deploying model to redis.

A simple webpage written in Svelte and a python webserver that:
  - Takes a search term
  - Scrapes instagram
  - Depupe using filters
  - Classify unique posts
  - Present discovered items in a grid.
  - Website keeps track of topk results on frontpage

