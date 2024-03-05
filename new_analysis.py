import spacy 
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_md") #uses the medium sized spacy analysis 
nlp.add_pipe('spacytextblob')

def cleaned_text(whole_reviwed_data):
    clean_review = {} #empty space to write values for sentiment abalysis
    for review in whole_reviwed_data:
        doc = nlp(review)
        clean_text = ' '.join(token.text.lower() for token in doc if not #uses tokens to remove stop words and seperates them into a list
                                token.is_stop and token.text.strip())
        clean_review[review] = clean_text #uses all the cleaned values and puts them into the empty text to review
    return(clean_review)

def sentiment_analysis(cleaned_data):
    for key, value in cleaned_data.items(): #uses cleaned values fror sentiment analysis
        doc = nlp(value)
        print(f"\nreview: {key}")
        print(f"sentiment:{doc._.blob.polarity}") #shows wther the review was positive or not

amazon_data = pd.read_csv("amazon_product_reviews.csv", sep = ",") #loads the data 
amazon_data = amazon_data.dropna(subset=["reviews.txt"]) #chooses the reviews collumn


sentiment_analysis(cleaned_text(amazon_data)) #does sentiment analysis on the reviews collumn
