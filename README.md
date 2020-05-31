# News-Headlines-Dataset-For-Sarcasm-Detection-Kaggle
News Headlines Dataset For Sarcasm Detection Kaggle

A very easy to understand and basic implementation to detect sarcasm in news headlines
** Kaggle link ** : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection

I kept the implementation very basic.

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 32, 16)            160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 6)                 102       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7         
=================================================================
Total params: 160,109
Trainable params: 160,109
Non-trainable params: 0
_________________________________________________________________
'''
You can add more layers in between as you see fit. This will provide you with a basic skeleton to start your work on.
