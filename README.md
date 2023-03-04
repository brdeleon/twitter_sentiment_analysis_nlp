# Leveraging NLP for Customer Sentiment Analysis on Twitter

## Uncovering Customer Sentiment to Improve Marketing and Boost Satisfaction

**Author**: [Brenda De Leon](mailto:brendardeleon@gmail.com)

## Overview

Twitter is one of the most popular and influential social networking platforms available today, with over 200 million monetizable daily active users [(mDAU)](https://www.statista.com/statistics/970920/monetizable-daily-active-twitter-users-worldwide/). Twitter's emphasis on real-time information allows for news to quickly and organically reach large populations. Twitter's quick and short message sharing allows for loose and easy engagement amongst all types of personalities, public figures, brands, and potential consumers.

We will use tweets and a sentiment classification model to create a better understanding of customer sentiment and provide actionable insights to increase customer satisfaction and strengthen marketing initiatives.

<img src="https://english.news.cn/20220316/aed3e20f331940c4b8c2b16c1f15b2e6/20220316aed3e20f331940c4b8c2b16c1f15b2e6_96e22deb2-5cd2-4a33-9335-c119411a9451.jpg.jpg" alt="Xinhua News Image of Product Launch" style="width: 550px;"/>

## Business Problem

Google's Marketing & Communications team wants to improve their social media communications strategy ahead of an exciting launch. Google's Marketing & Communications team has contracted us to help develop a data backed tool to help with their social media communications strategy.

Google wants a tool to help with: 
1. Improving Customer Satisfaction 
2. Improving Marketing Campaigns

Unfortunately, classifying tweets and mining for tweets with only hashtags can be tedious and costly. We will build a model that can efficiently classify tweets as positive, negative, or neutral sentiment. The sentiment classification will also allow us to produce WordClouds that display the top words and phrases associated with each sentiment regarding the brand or product and bar graphs of the top hashtags represented in each sentiment. By classifying tweets into sentiment, Google will be able to extract patterns with more nuance. 

## Data

The dataset contains tweets from a popular annual conference known to showcase innovation called South by South West ("SXSW") and were identified using the hashtag: #SXSW. The tweets were gathered by CrowdFlower and were evaluated and labeled for sentiment through crowdsource. Contributors were also asked to say which brand or product was the target of that sentiment. The dataset contains over 9000 tweets and can be found [here](https://data.world/crowdflower/brands-and-product-emotions). From the provided data, we engineered additional features.

## Methods

1. Load the Data

2. Perform Data Cleaning and Exploratory Data Analysis with nltk
Our data preprocessing will include removing: punctuation, special characters, and stop words, as well as converting all text to lowercase and lemmatizing. We will compare the raw word frequency distributions of each category.

3. Build and Evaluate a Baseline Model 
Ultimately all data must be in numeric form in order to be able to fit a scikit-learn model. We'll use a sklearn tool, TfidfVectorizer, to convert all tweet text data into a vectorized format.

4. Iteratively Perform and Evaluate Preprocessing and Feature Engineering Techniques
Investigate different algorithms and techniques to determine whether they should be part the final model.

5. Evaluate a Final Model on the Test Set.

## Results

We built a <b>Logistic Regression</b> model that is able to classify a tweet based on sentiment.  

We can determine the sentiment of a tweet based on engineered features at an accuracy of 88% and at an f1 macro score of 73%.

![final model](/finalmodel.png)

`------------------------------------------------------------`<br>
`Logistic Regression(Final Model) CLASSIFICATION REPORT TESTING` <br>
`------------------------------------------------------------`<br>
`              precision    recall  f1-score   support`<br>
<br>
`    Negative       0.59      0.29      0.39       148`<br>
`     Neutral       0.93      0.97      0.95      1126`<br>
`    Positive       0.84      0.87      0.85       706`<br>
<br>
`    accuracy                           0.88      1980`<br>
`   macro avg       0.79      0.71      0.73      1980`<br>
`weighted avg       0.87      0.88      0.87      1980`<br>

We have determined which features are most important in classifying sentiment:

`brand   0.307 +/- 0.011` <br>
`clean_tweets 0.060 +/- 0.002` <br>
`stopword_count 0.016 +/- 0.004` <br>
`hashtags 0.014 +/- 0.002` <br>
`avg_wordlength 0.013 +/- 0.004` <br>
`stopwords_vs_words 0.011 +/- 0.003` <br>
`sent_count 0.010 +/- 0.005` <br>
`unique_word_count 0.007 +/- 0.003` <br>
`punct_count 0.003 +/- 0.001`

## Conclusions

Google can use this classification model to identify the emotion of tweets about a particular topic, the topic could be past launches, new products, or the brand itself. With Google's upcoming launch, Google can analyze the words, phrases, and hashtags of past launches by sentiment to better understand the audience's reception to the launch to help shape the strategy for the new launch. Equally, Google can use this model during the launch for real time feedback and after the launch to analyze for feedback. By classifying tweets into sentiment classes, Google will be able to extract more meaningful patterns with the help of word clouds and graphs. 


![google negative cloud](/google%20negative%20word%20cloud.png)

<br>

![google positive graph](/googlepositivehashtag.png)

### Next Steps

 - Better data collection could significantly improve our prediction ability. We have an imbalanced dataset with majority "Neutral" sentiment values. More data, particularly for the minority classes could improve the model's performance. Additionally, some of the tweets were mislabeled, for next steps our model could benefit from training on more accurate labeled tweets.
 - Include new data by web scraping tweets so that we are able to collect usernames of tweet poster and so that our model is able to train on newer and larger data.
 - Use specific tweet tokenizer so that our model is able to handle emojis.
 - Engineer additional features like assigning a sentiment intensity score to each tweet using nltk' vader package.
 - Given the high weight the random forest algorithm gives to the hashtags feature, we should further inspect it for patterns.
 - Use feature importance to improve our model, removing those features with lower scores.

## For More Information

See the full analysis in the [Jupyter Notebook](</Tweet_Sentiment_Modeling.ipynb>) or review this [presentation](</Twiter_Sentiment_Presentation.pdf>).

For additional info, contact Brenda De Leon at [brendardeleon@gmail.com](mailto:brendardeleon@gmail.com)

## Repository Structure

```
├── data
│   ├── judge_1377884607_tweet_product_company.csv
│   └── clean_df.csv
├── Tweet_Sentiment_Analysis.ipynb
├── Tweet_Sentiment_EDA.ipynb
├── Tweet_Sentiment_Modeling.ipynb
├── Twiter Sentiment Presentation.pdf
└── README.md
```