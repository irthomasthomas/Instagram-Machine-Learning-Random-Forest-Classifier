This project was a learning exercise. An exploration of machine-learning through the training and deployment of a Random Forest Classifier.

The project scales the walled-garden of Instagram to find posts containing items for sale from instagram citizens, and make them searchable. It was conceived of at a time when ML was still mostly for researchers and big tech. Tools for continous training and deployment where few. I used a brand-new stack of tools for ML by Redis Labs. Being alpha software it changed frequently. Using this project, now, would probably require significant re-factoring. 

I manually classified thousands of posts, and used tools to classify many more.

The main tools used where
RedisAI
RedisGears
SciKit
Svelte JS
etc

RedisBloom and various probabilistic data-structures where used to support a public web interface. The probabilistic data-structures allowed me to run a fully-functional public demo on a single small VM in a deterministic manner.

Briefly, the program consists of:

  Scraping instagram posts from hashtags related selling.
  
  Manually tagging posts as relevant or not.
  
  Passing the post text through a text processing pipeline in python running on RedisGears:
  
   Cleaning
   
   Tokenizing
   
   etc
   
  Training a Random-Forest-Classifier on the cleaned text.
  Converting the RFC to matrix
  Deploying model to redis.

A simple webpage written in Svelte that:
  Takes a search term 
  Scrapes instagram
  Depupe using filters
  Classify unique posts
  Present discovered items in a grid.
  Website keeps track of topk results on frontpage

