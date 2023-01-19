# Twitter Sentiment Analysis

![Launch Event](https://english.news.cn/20220316/aed3e20f331940c4b8c2b16c1f15b2e6/20220316aed3e20f331940c4b8c2b16c1f15b2e6_96e22deb2-5cd2-4a33-9335-c119411a9451.jpg.jpg)



![Launch Event](https://static01.nyt.com/images/2017/11/08/business/08TWITTER1/08TWITTER1-superJumbo.jpg?quality=75&auto=webp)

# Social Media Marketing

**Author**: [Brenda De Leon](mailto:brendardeleon@gmail.com)

## Overview

Twitter is one of the most popular and influential social networking platforms available today, with over 200 million monetizable daily active users [(mDAU)](). Twitter users can send short 280-character messages called tweets to their followers. Twitter users can follow other users, as well as read and share their tweets. Twitter's emphasis on real-time information allows for news to quickly and organically reach large populations.



(one or two sentences about why tweets or sentiment analysis intro) A study held in 2022 revealed that Twitter users regularly use Twitter for news.(2022)
Researchers found that of nearly 10 million clicks in a random sample of news stories posted on Twitter, 61% of the clicks


Unfortunately, the city resources available to address traffic safety are limited. 

We will use tweets and their sentiment classification to get insights to improve customer satisfaction and marketing campaigns.
Data Source¶

We, the Vision Zero initiative, are committed to working with the City of Chicago to eliminate fatalities and serious injuries from traffic crashes. 

---

Traffic deaths are preventable. The City of Chicago has had at least 800,000 yearly reported crash incidents since 2017 and at least 100 yearly reported crash incidents with at least one fatality since 2018. The City of Chicago believes everyone has the right to access to safe streets. However, Chicago has seen a general increase in incidents with at least one fatality over the years even though there has been a general decrease in number of crash incidents. 


## Business Problem

Google's Marketing & Communications team wants to improve their social media communications strategy. For fresh eyes, Google's Marketing & Communications team has decided to outsource and contract us to help develop a data backed social media communications strategy for an upcoming launch. Google wants help with:

 1. Improving customer satisfaction
 2. Improving marketing campaigns


## Data

Our dataset contains tweets following a Google and Apple product launch at a SXSW event, the tweets were gathered by CrowdFlower and were evaluated and labeled for sentiment through crowdsource. Contributors were also asked to say which brand or product was the target of that sentiment. The dataset contains over 9000 rows and be found [here](https://data.world/crowdflower/brands-and-product-emotions). From the provided data, we engineered additional features.

## Methods



We will use the data to create a sentiment classifying machine learning model. Our multiclass classification model will help identify what features are most relevant in sentiment analysis. Our data cleaning process will include removing: punctuation, special characters, and numbers, as well as converting all text to lowercase. Cleaning and normalizing the text prevents our models from being influenced by minor variations. In order for our models to interpret the tweets, we transform the text data into numerical features through a vectorizer. 




"In language processing,
the vectors x are derived from textual data,
in order to reflect various linguistic properties of the text."

- Yoav Goldberg

## Results

We built a <b>logistic regression</b> model that is able to classify a tweet based on sentiment.  

We can determine the sentiment of a tweet based on engineered features at an accuracy of 88% and at an f1 macro score of 75%. We have determined which features are most important in classifying sentiment:

`FIRST_CRASH_TYPE0.389 +/- 0.004` <br> 
`LOCATION0.267 +/- 0.001` <br>
`TRAFFICWAY_TYPE0.183 +/- 0.002` <br>
`CRASH_MONTH0.107 +/- 0.002` <br>
`CRASH_HOUR0.103 +/- 0.002` <br>
`CRASH_DAY_OF_WEEK0.097 +/- 0.002` <br>
`TRAFFIC_CONTROL_DEVICE0.086 +/- 0.002` <br>
`DEVICE_CONDITION0.080 +/- 0.002` <br>
`LIGHTING_CONDITION0.073 +/- 0.002` <br>
`SPEED_LIMIT0.048 +/- 0.001` <br>
`ROADWAY_SURFACE_COND0.043 +/- 0.001` <br>
`WEATHER_CONDITION0.025 +/- 0.001` <br>
`VEHICLE_TYPE0.015 +/- 0.001`

![final model](/models.png)


## Conclusions

to use this predictive model to identify the emotion of tweets about their conference. Using the predicted emotions, South by Southwest can analyze the words and phrases asscociated with the emotions to better understand audience reception and anticipation of the event.

Our recommendations for Google's Marketing & Communications team to improve their social media communications strategy
are backed by data and focus, recommends the following:

- **Improving Customer Satisfaction** 

reaching out directly
retweets


- **Improving Marketing Campaigns** 

popular positive
several days

competitiors, what is working 
similarly what to avoid

hashtags


### Next Steps

Further analyses could yield additional insights to further improve insight quality:

- Better data collection could significantly improve our prediction ability. Unfortunately, features worthy of attention were dropped due to excessive nulls or unknowns. Age and sex are some features that were dropped due to having almost 90% null values. Weather and safety equipment are other possibly important features that were dropped to due to excessive nulls or unknowns.
- Widen the crash date range to include crashes from more years.
- Use location points to create crash incident maps to identify if there are any top locations that cluster near each other for priority identification.
- Conduct a more extensive grid search on random forest focusing on those parameters that can address our data imbalance like class weight.

## For More Information

See the full analysis in the [Jupyter Notebook](</Tweet_Sentiment_Modeling.ipynb>) or review this [presentation](</Twiter Sentiment Presentation.pdf>).

For additional info, contact Brenda De Leon at [brendardeleon@gmail.com](mailto:brendardeleon@gmail.com)

<img src="https://activetrans.org/busreports/wp-content/uploads/2015/04/vision_zero_logo.jpg" alt="visionlogo" style="width: 200px;"/>

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

###### Requirements

1. Load the Data
Use pandas and sklearn.datasets to load the train and test data into appropriate data structures. Then get a sense of what is in this dataset by visually inspecting some samples.

2. Perform Data Cleaning and Exploratory Data Analysis with nltk
Standardize the case of the data and use a tokenizer to convert the full posts into lists of individual words. Then compare the raw word frequency distributions of each category.

3. Build and Evaluate a Baseline Model with TfidfVectorizer and MultinomialNB
Ultimately all data must be in numeric form in order to be able to fit a scikit-learn model. So we'll use a tool from sklearn.feature_extraction.text to convert all data into a vectorized format.

Initially we'll keep all of the default parameters for both the vectorizer and the model, in order to develop a baseline score.

4. Iteratively Perform and Evaluate Preprocessing and Feature Engineering Techniques
Here you will investigate three techniques, to determine whether they should be part of our final modeling process:

Removing stopwords
Using custom tokens
Domain-specific feature engineering
Increasing max_features
5. Evaluate a Final Model on the Test Set
Once you have chosen a final modeling process, fit it on the full training data and evaluate it on the test data.



## Modeling

We will iteratively perform and evaluate preprocessing and feature engineering techniques. We will investigate different techniques to determine whether they should be part of our final modeling process:

Domain-specific feature engineering
Increasing max_features
Balancing class weight



The data needs to be able to fit a scikit-learn model. We will standardize the case of the data, use a tokenizer to convert the full tweets into lists of individual words. We will then compare the raw word frequency distributions of each sentiment. 
