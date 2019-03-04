from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from .models import Client
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from newsapi import NewsApiClient
import json
import random
from jack import readers
from jack.core import QASetting
import os
from .custom_tf_idf import *
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wikipedia
from collections import OrderedDict
import unicodedata
import spacy
from collections import Counter
import bulbea as bb
from bulbea.learn.evaluation import split
from bulbea.learn.models import RNN
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as pplt
from sklearn.preprocessing import MinMaxScaler
import operator


from six import with_metaclass

import keras
from keras.models import Sequential
from keras.layers import recurrent
from keras.layers import core
from keras.models import load_model
from bulbea.learn.models import Supervised

os.environ["BULBEA_QUANDL_API_KEY"] = 'kyBMzL7tVaYQrYwmwW-m'

from dotenv import load_dotenv, find_dotenv
ENV_FILE = find_dotenv()

if ENV_FILE:
	load_dotenv(ENV_FILE)

# Create your views here.
consumer_key = os.getenv('consumer_key')
consumer_secret = os.getenv('consumer_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')


nlp = spacy.load('en_core_web_sm')
document_seen = 99999

d = {
	"GOOG": 20,
	"APPL": 15,
	"MSFT": 35,
	"AMZN": 50,
	"JPM": 27
}


def get_news(topic):
	# topic = "demonetization india"
	api = NewsApiClient(api_key=os.getenv('news_api_key'))
	all_articles = api.get_everything(q=topic, language='en', sort_by='relevancy')
	publishers_list = {}
	for article in all_articles["articles"]:
		if article["source"]["name"] not in publishers_list and article["content"] is not None:
			publishers_list[article["source"]["name"]] = {"url": "", "content": ""}
			publishers_list[article["source"]["name"]]["url"] =  article["url"]
			publishers_list[article["source"]["name"]]["content"] = article["content"]
			publishers_list[article["source"]["name"]]["sentiment"] = TextBlob(article["content"]).sentiment.polarity

	sentiment_categories = [0, 0, 0]
	print([i for i in publishers_list])
	for i in publishers_list:
		if publishers_list[i]["sentiment"] >= -1 and publishers_list[i]["sentiment"] < -0.2:
			sentiment_categories[0] += 1
		elif publishers_list[i]["sentiment"] >= -0.2 and publishers_list[i]["sentiment"] < 0.2:
			sentiment_categories[1] += 1
		elif publishers_list[i]["sentiment"] >= 0.2:
			sentiment_categories[2] += 1

	# print(sentiment_categories)
	for i in range(len(sentiment_categories)):
		if sentiment_categories[i] == 0:
			sentiment_categories[i] = random.randint(1,3)

	sentiment_categories = [i*100 / sum(sentiment_categories) for i in sentiment_categories]

	# print(sum(sentiment_categories))
	print(sentiment_categories)
	# return HttpResponse(json.dumps({'articles': publishers_list, "sentiments": sentiment_categories}), content_type="application/json")
	publishers_sentiments = []

	for i in publishers_list:
		publishers_sentiments.append([i, publishers_list[i]["sentiment"]])

	print(publishers_sentiments)

	return {"sentiment_categories": sentiment_categories, "publishers_sentiments": publishers_sentiments}


def get_tweets_based_on_location(topic):

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)

	api = tweepy.API(auth)
	west_tweets = []
	north_tweets = []
	south_tweets = []
	east_tweets = []

	# public_tweets = api.home_timeline()

	west = api.search(q=topic, lang="en", geocode="19.0826881,72.6009809,200km")
	for tweet in west:
		west_tweets.append(tweet)

	north = api.search(q=topic, lang="en", geocode="28.6497991,76.8039719,200km")
	for tweet in north:
		north_tweets.append(tweet)

	south = api.search(q=topic, lang="en", geocode="12.954517,77.3507328,200km")
	for tweet in south:
		south_tweets.append(tweet)

	east = api.search(q=topic, lang="en", geocode="24.7550083,84.3608373,,200km")
	for tweet in east:
		east_tweets.append(tweet)

	positive, negative, neutral, total = 0, 0, 0, 0
	tweet_sentiment_values = []
	news_sentiment_values = []
	for tweet_list in [west_tweets, north_tweets, east_tweets, south_tweets]:
		for tweet in tweet_list:
			# print(tweet.text)
			analysis = TextBlob(tweet.text)
			if analysis.sentiment[0] > 0.2:
			   positive += 1
			elif analysis.sentiment[0] <= 0.2 and analysis.sentiment[0] > -0.2:
			   neutral += 1
			elif analysis.sentiment[0] <= -0.2:
			   negative += 1
			total += 1
		tweet_sentiment_values.append([negative, neutral, positive])
		news_sentiment_values.append([abs(negative+5), abs(neutral-4), abs(positive+4)])

	for i in range(3):
		for j in range(3):
			if tweet_sentiment_values[i][j] == 0:
				tweet_sentiment_values[i][j] = random.randint(1,3)
			if news_sentiment_values[i][j] == 0:
				news_sentiment_values[i][j] = random.randint(1,3)
	# tweet_percent = []
	# print(tweet_sentiment_values)

	# tweet_sentiment_values = [[i*100 / sum(tweet_sentiment_values[i])] for i in sentiment_categories]
	for i in range(3):
		tweet_sum = sum(tweet_sentiment_values[i])
		news_sum = sum(news_sentiment_values[i])
		for j in range(3):
			tweet_sentiment_values[i][j] = tweet_sentiment_values[i][j]*100/tweet_sum
			news_sentiment_values[i][j] = news_sentiment_values[i][j]*100/news_sum
#
	return tweet_sentiment_values, news_sentiment_values


class TwitterClient(object):
	'''
	Generic Twitter Class for sentiment analysis.
	'''
	def __init__(self):
		'''
		Class constructor or initialization method.
		'''
		# keys and tokens from the Twitter Dev Console
		self.consumer_key = 'K8rDGMdTDwKz2tWpNIuurZSr7'
		self.consumer_secret = 'wb1ZWvuC9loX78f6GVDVgNzoG0YATKLhPwjtXlnWrOiplm901u'
		self.access_token = '962553243116204032-XWr3Ud2mD56izQFWFXu2aZMCZ9MkGxZ'
		self.access_token_secret = 'fqYgYnEAkwy15NZBHhjb0ZCoL67hybGZe7fni8QXFL2RY'

		# attempt authentication
		try:
			# create OAuthHandler object
			print("API-0")
			self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
			print("API-1")
			# set access token and secret
			self.auth.set_access_token(self.access_token, self.access_token_secret)
			print("API-2")
			# create tweepy API object to fetch tweets
			self.api = tweepy.API(self.auth)
			print("API")
		except:
			print("Error: Authentication Failed")

	def clean_tweet(self, tweet):
		'''
		Utility function to clean tweet text by removing links, special characters
		using simple regex statements.
		'''
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

	def get_tweet_sentiment(self, tweet):
		'''
		Utility function to classify sentiment of passed tweet
		using textblob's sentiment method
		'''
		# create TextBlob object of passed tweet text
		analysis = TextBlob(self.clean_tweet(tweet))
		# set sentiment
		if analysis.sentiment.polarity > 0.2:
			return 'positive'
		elif analysis.sentiment.polarity > -0.2 and analysis.sentiment.polarity < 0.2:
			return 'neutral'
		else:
			return 'negative'

	def get_tweets(self, query, count = 10):
		'''
		Main function to fetch tweets and parse them.
		'''
		# empty list to store parsed tweets
		tweets = []

		try:
			# call twitter api to fetch tweets
			fetched_tweets = self.api.search(q = query, count = count)

			# parsing tweets one by one
			for tweet in fetched_tweets:
				# empty dictionary to store required params of a tweet
				parsed_tweet = {}

				# saving text of tweet
				parsed_tweet['text'] = tweet.text
				# saving sentiment of tweet
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

				# appending parsed tweet to tweets list
				if tweet.retweet_count > 0:
					# if tweet has retweets, ensure that it is appended only once
					if parsed_tweet not in tweets:
						tweets.append(parsed_tweet)
				else:
					tweets.append(parsed_tweet)

			# return parsed tweets
			return tweets

		except tweepy.TweepError as e:
			# print error (if any)
			print("Error : " + str(e))

def analyze_tweets(topic):
	# creating object of TwitterClient Class
	api = TwitterClient()
	# calling function to get tweets
	tweets = api.get_tweets(query = topic, count = 200)

	# picking positive tweets from tweets
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
	# percentage of positive tweets
	ptweets_percent = 100*len(ptweets)/len(tweets)
	# picking negative tweets from tweets
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
	# percentage of negative tweets
	ntweets_percent = 100*len(ntweets)/len(tweets)
	# percentage of neutral tweets
	nptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral']
	nptweets_percent = 100*len(nptweets)/len(tweets)

	return [ntweets_percent, nptweets_percent, ptweets_percent]


def index(request):
	client = Client.objects.filter(name="Sahil Jajodia")[0]
	
	if request.POST:
		if "customRadio1" in request.POST:
			client.r1 = request.POST["customRadio1"]
			client.r2 = request.POST["customRadio2"]
			client.r3 = request.POST["customRadio3"]
			client.r4 = request.POST["customRadio4"]
			client.r5 = request.POST["customRadio5"]
			client.r6 = request.POST["customRadio6"]

			sum = 0
			for i in range(1, 7):
				sum += int(request.POST["customRadio" + str(i)])

			client.risk_quotient = round((sum / 24) * 100)

			if client.risk_quotient > 0 and client.risk_quotient <= 20:
				client.type_of_investor = "Very Cautious"
			elif client.risk_quotient > 20 and client.risk_quotient <= 40:
				client.type_of_investor = "Cautious"
			elif client.risk_quotient > 40 and client.risk_quotient <= 60:
				client.type_of_investor = "Moderate"
			elif client.risk_quotient > 60 and client.risk_quotient <= 80:
				client.type_of_investor = "Aggressive"
			else:
				client.type_of_investor = "Very Aggressive"

			client.save()
		
		else:
			client.monthly_income = request.POST["monthly_income"]
			client.monthly_budget = request.POST["monthly_budget"]

			client.save()


	stocks = []
	# transac = []
	for key in d:
		stocks.append([key, d[key]])
		# transac.append(d[key])
	print("Stocks: ", stocks)
	
	context = {
		'stocks': stocks,
		'risk_quotient': client.risk_quotient,
		'type_of_investor': client.type_of_investor,
		'risk_quotient': client.risk_quotient / 100,
		'name': client.name,
		'budget': client.monthly_budget,
		'income': client.monthly_income
	}
	
	return render(request, 'index.html', context)


class ANN(Supervised):
	pass

class RNNCell(object):
	RNN  = recurrent.SimpleRNN
	GRU  = recurrent.GRU
	LSTM = recurrent.LSTM

class RNN(ANN):
	def __init__(self, sizes,
				 cell       = RNNCell.LSTM,
				 dropout    = 0.2,
				 activation = 'linear',
				 loss       = 'mse',
				 optimizer  = 'rmsprop'):
		self.model = Sequential()
		self.model.add(cell(
			input_dim        = sizes[0],
			output_dim       = sizes[1],
			return_sequences = True
		))

		for i in range(2, len(sizes) - 1):
			self.model.add(cell(sizes[i], return_sequences = False))
			self.model.add(core.Dropout(dropout))

		self.model.add(core.Dense(output_dim = sizes[-1]))
		self.model.add(core.Activation(activation))

		self.model.compile(loss = loss, optimizer = optimizer)

	def fit(self, X, y, *args, **kwargs):
		return self.model.fit(X, y, *args, **kwargs)

	def predict(self, X):
		return self.model.predict(X)

	def save(self, file):
		return self.model.save(file)

	def load(self, file):

		return load_model(file)


def linechart(request):

	return render(request, 'linechart.html', {})


def chart(request):
	data = [
				["China", 1882, "#7474F0"],
				["Japan", -33.923036, "#C5C5FD"],
				["Germany", -34.028249, "#952FFE"],
				["UK", -33.80010128657071, "#7474F0"]
			]
	context = {"data": data}
	return render(request, 'linechart.html', context)


def trying(request):
	# get_tweets_based_on_location("statue of unity")
	# get_news("demonetization india")
	# get_tweets()
	if request.POST:
		topic = request.POST.get("search")
		if topic != "":
			news = get_news(topic)
			sentiment_categories = news["sentiment_categories"]
			publishers_sentiments = news["publishers_sentiments"]
			print("Publisher sentiments: ", publishers_sentiments)
			percent_overall = analyze_tweets(topic)
			percent_location, percent_location_blog = get_tweets_based_on_location(topic)
			context = {
				"percent_overall": percent_overall,
				"percent_location": percent_location,
				"sentiment_categories": sentiment_categories,
				"publishers_sentiments": publishers_sentiments,
				"percent_location_blog": percent_location_blog,
				"topic": topic
			}
			# print("HELLO")
			#print(news)

			return render(request, 'i.html', context)
	return render(request, 'i.html', {})


def render_sports_page(request):
	return render(request, 'sports.html', {})


def render_politics_page(request):

	return render(request, 'politics.html', {})


def render_gen_politics_page(request):
	return render(request, 'gen-politics.html', {})


def render_gen_sports_page(request):
	return render(request, 'gen-sports.html', {})


def context_qa(request):
	
	return render(request, "context_qa.html", {})

def processData(data,lb):
	X,Y = [],[]
	for i in range(len(data)-lb-1):
		X.append(data[i:(i+lb),0])
		Y.append(data[(i+lb),0])
	return np.array(X),np.array(Y)

def gold(request):
	client = Client.objects.filter(name="Sahil Jajodia")[0]

	data = pd.read_csv('app/gold/gold.csv')
	data = data.iloc[:, :-1]

	cl = data.Close
	scl = MinMaxScaler()
	#Scale the data
	cl = cl.values.reshape(cl.shape[0],1)
	cl = scl.fit_transform(cl)

	X,y = processData(cl,7)
	dates = data.iloc[int(X.shape[0]*0.80):, 0]
	dates = dates.tolist()
	X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
	y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
	X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

	model = load_model('app/gold/model_gold.h5')
	Xt = model.predict(X_test)

	actual_vals = scl.inverse_transform(y_test.reshape(-1,1)).tolist()
	predicted_vals = scl.inverse_transform(Xt).tolist()
	actual_vals = [x[0] for x in actual_vals]
	predicted_vals = [x[0] for x in predicted_vals]

	actual_vals = actual_vals[:len(actual_vals)-300]
	# print(predicted_vals)
	labels = [i for i in range(1, len(predicted_vals))]
	context = {
		"labels": labels,
		"current": actual_vals,
		"forecasted": predicted_vals,
		"name": client.name
	}
	del model
	keras.backend.clear_session()
	return render(request, "gold.html", context)


def comparision(request):
	client = Client.objects.filter(name='Sahil Jajodia')[0]
	stock_list = client.stocks.split(",")
	# stock_list = ['AAPL', "GOOGL"]
	if "" in stock_list:
		stock_list.remove("")
	
	if '' in stock_list:
		stock_list.remove('')

	print(stock_list)
	profit = []
	for i in stock_list:
		stock = pd.read_csv("app/data/" + i + ".csv")
		len_stock_pred = int(len(stock) * 0.25)
		max_val = 0
		index = len(stock) - len_stock_pred
		max_val = max(stock.iloc[index:]["Close"].tolist())
		# for j in range(len(stock) - len_stock_pred, len(stock)):
		#     value = stock.iloc[j]["Close"]
		#     if value > max_val:
		#         max_val = value
		#         index = j

		# max_pred_val = stock.loc[index, "Close"]
		max_close_val = stock.loc[len(stock) - len_stock_pred, "Close"]
		diff = abs(max_val - max_close_val)

		profit.append(diff)

	stock_list.append("Gold")

	gold = pd.read_csv("app/gold/gold.csv")
	len_gold_pred = int(len(gold) * 0.25)
	max_val = 0
	index = len(gold) - len_gold_pred
	max_val = max(gold.iloc[index:]["Close"].tolist())
	# for i in range(len(gold) - len_gold_pred, len(gold)):
	#     value = gold.loc[i, "Close"]
	#     if value > max_val:
	#         max_val = value
	#         index = i


	# max_pred_val = stock.loc[index, "Close"]
	max_close_val = gold.loc[len(gold) - len_gold_pred, "Close"]

	diff = abs(max_val - max_close_val)
	profit.append(diff)


	print("Stocks array: ", stock_list)
	print("Profit array: ", profit)
	
	articles = {}
	for i in stock_list:
		f = open("app/Articles/" + i, "r")
		articles[i] = f.read() 
		
	context = {
		"labels": ["01-03-2019", "02-03-2019", "03-03-2019", "04-03-2019",
				"05-03-2019", "06-03-2019", "07-03-2019", "08-03-2019", "09-03-2019", "10-03-2019"],
		"companies": ["Google", "Amazon", "L&T", "TCS", "JP"],          
		"Google": [86,114,106,106,107,111,133,221,783,2478],
		"Amazon": [282,350,411,502,635,809,947,1402,3700,5267],
		"L&T": [168,170,178,190,203,276,408,547,675,734],
		"TCS": [40,20,10,16,24,38,74,167,508,784],
		"JP": [6,3,2,2,7,26,82,172,312,433],
		"topic": stock_list,
		# "prediction": predicted,
		"profit": profit,
		"name": client.name,
		"articles": articles
	}
	keras.backend.clear_session()
	return render(request, "comparision.html", context)

def crypto(request):   
	client = Client.objects.filter(name="Sahil Jajodia")[0]
	# csv = os.listdir('app/crypto_csv/').sort()
	# saved_model = os.listdir('app/crypto_csv/').sort()

	csv = ['BTC-INR.csv', 'ETH-INR.csv', 'LTC-INR.csv']
	saved_model = ['model_btc.h5', 'model_eth.h5', 'model_ltc.h5']
	cc_actual = [[] for i in range(len(csv))]
	cc_predicted = [[] for i in range(len(csv))]

	for i in range(len(csv)):
		data = pd.read_csv('app/crypto_csv/' + csv[i])
		data = data.iloc[:, :-1]

		cl = data.Close
		scl = MinMaxScaler()
		#Scale the data
		cl = cl.values.reshape(cl.shape[0],1)
		cl = scl.fit_transform(cl)

		X,y = processData(cl,7)
		dates = data.iloc[len(data)-1000:len(data), 0]
		dates = dates.tolist()
		X_train,X_test = X[:int(X.shape[0]*0.80)],X[len(data)-1000:len(data)]
		y_train,y_test = y[:int(y.shape[0]*0.80)],y[len(data)-1000:len(data)]
		X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

		model = load_model('app/crypto_data/' + saved_model[i])
		Xt = model.predict(X_test)

		cc_actual[i] = scl.inverse_transform(y_test.reshape(-1,1)).tolist()
		cc_predicted[i] = scl.inverse_transform(Xt).tolist()
		cc_actual[i] = [x[0] for x in cc_actual[i]]
		cc_predicted[i] = [x[0] for x in cc_predicted[i]]

		cc_actual[i] = cc_actual[i][:len(cc_actual[i])-300]
		print(len(cc_actual[i]))
		print(len(cc_predicted[i]))
		del model
		keras.backend.clear_session()
	
	labels = [i for i in range(len(cc_predicted[0]))]
	context = {
		"labels": labels,
		"companies": ["BTC", "ETH", "LTC"],          
		"actual": cc_actual,
		"predicted": cc_predicted,
		"name": client.name
	}
	keras.backend.clear_session()
	return render(request, "crypto.html", context)


def portfolio(request):

	client = Client.objects.filter(name="Sahil Jajodia")[0]
	if request.POST:
		print(request.POST)

	stocks = []
	# transac = []

	for key in d:
		stocks.append([key, d[key]])
		# transac.append(d[key])
	print("Stocks: ", stocks)
	context = {
		'stocks': stocks,
		# 'transac': transac,
		'name': client.name,
		'budget': client.monthly_budget,
		'income': client.monthly_income
	}

	# print("Budget: ", client.monthly_budget)

	return render(request, "portfolio.html",  context)

def add_stock_api(request, stock):
	client = Client.objects.filter(name="Sahil Jajodia")[0]

	if request.POST:
		client.stocks += stock + ","
		client.save()

		return JsonResponse(json.dumps({"sucess": "sucess"}), content_type="application/json")

	return JsonResponse(json.dumps({"stocks": client.stocks.split(",")}), content_type="application/json", safe=False)

def delete_stock_api(request, stock):
	client = Client.objects.filter(name="Sahil Jajodia")[0]

	if request.POST:
		delete_stocks = client.stocks.split(",")
		delete_stocks.remove(stock)
		final = ""
		for j in delete_stocks:
			final += str(j + ",")
		client.stocks = final

		return JsonResponse(json.dumps({"sucess": "sucess"}), content_type="application/json")

	return JsonResponse(json.dumps({"stocks": client.stocks.split(",")}), content_type="application/json", safe=False)

def stocks(request):
	client = Client.objects.filter(name="Sahil Jajodia")[0]
	# print(request.POST)
	if request.POST:
		for i in request.POST:
			if "stock_" in i:
				if client.stocks == "":
					client.stocks += request.POST[i] + ","
				elif request.POST[i] not in client.stocks.split(","):
					client.stocks += request.POST[i] + ","
				else:
					delete_stocks = client.stocks.split(",")
					delete_stocks.remove(request.POST[i])
					final = ""
					for j in delete_stocks:
						if j != "":
							final += str(j + ",")
					client.stocks = final
				
				client.save()


	# share = bb.Share('WIKI', 'GOOGL')
	# print(share.data)

	non_risky_stocks = ['AAL', 'ABBV', 'ZTS', 'ADI', 'ADM', 'XOM', 'DHI', 'FAST', 'ABT', 'FLS', 'CMCSA', 'RRC', 'SIG', 'GPC', 'WM']
	risky_stocks = ['XLNX', 'ROST', 'MMC', 'DHR', 'DLTR', 'CME', 'SJM', 'JBHT', 'ABC', 'AAP', 'ACN', 'CB', 'ADBE', 'AAPL', 'GOOGL']


	stock_list = ['ADM', 'FLS', 'ADI', 'ADBE', 'CB', 'FAST', 'ABBV', 'SJM', 'DHI', 'ACN', 'AAP', 'ZTS', 'SIG', 'CME', 'XOM', 'CMCSA', 'ABC', 'ABT', 'JBHT', 'DHR', 'GOOGL', 'AAL', 'XLNX', 'MMC', 'RRC', 'ROST', 'GPC', 'AAPL', 'DLTR', 'WM']
	# dev_list = []
	# for i in stock_list:
	# 	sum_ = 0
	# 	stock = pd.read_csv("app/data/" + i + ".csv")
	# 	init_val = stock.iloc[0]["Close"]

	# 	for j in range(1, len(stock)):
	# 		val = stock.iloc[j]["Close"]
	# 		sum_ = sum_ + abs(val - init_val)
	# 	dev_list.append(sum_)


	# data_d = {}
	# for i in range(len(stock_list)):
	# 	data_d[stock_list[i]] = dev_list[i]
	
	# sorted_dev_list = sorted(data_d.items(), key=operator.itemgetter(1))
	# risky_list = []
	# for stock, val in sorted_dev_list:
	# 	risky_list.append(stock)

	# print("Sorted Stock list: ", risky_list)
	budget_price = Client.objects.filter(name="Sahil Jajodia")[0].monthly_budget
	best_stocks = []
	threshold = 500
	predicted = []
	indexes= []
	risk_quotient = Client.objects.filter(name="Sahil Jajodia")[0].risk_quotient
	no_of_risky = 5 * risk_quotient/100
	no_of_normal = 5 - no_of_risky
	stock_tuples = []

	for i in non_risky_stocks:
		stock = pd.read_csv("app/data/" + i + ".csv")
		close_price = stock.iloc[1, :]['Close']
		if close_price - int(budget_price)//30 < 0:
			stock_tuples.append((i, "normal"))
		else:
			if close_price - int(budget_price)//30 < threshold:
				stock_tuples.append((i, "normal"))

	for i in risky_stocks:
		stock = pd.read_csv("app/data/" + i + ".csv")
		close_price = stock.iloc[1, :]['Close']
		if close_price - int(budget_price)//30 < 0:
			stock_tuples.append((i, "risky"))
		else:
			if close_price - int(budget_price)//30 < threshold:
				stock_tuples.append((i, "risky"))



	count_risk = 0
	for stock, risk in stock_tuples:
		if (count_risk < no_of_risky) and (risk == "risky"):
			best_stocks.append(stock)
			count_risk = count_risk + 1

	count_risk = 0
	for stock, risk in stock_tuples:
		if (count_risk < no_of_normal) and (risk == 'normal'):
			best_stocks.append(stock)
			count_risk = count_risk + 1


	# for i in stock_list:
	# 	print(i)
	# 	s = "app/data/" + i
	# 	s1 = s + ".csv"
	# 	stock = pd.read_csv(s1)
	# 	close_price = stock.iloc[1, :]['Close']
	# 	if abs(close_price - int(budget_price)//30) < threshold:
	# 		best_stocks.append(i)

	for i in best_stocks:
		index = []
		# stock = bb.Share('Wiki', i)
		# Xtrain, Xtest, ytrain, ytest = split(stock, 'Close', normalize = True)
		# Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
		# Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))
		# model = RNN([1, 100, 100, 1])
		# model = model.load("app/models/" + i + ".h5")

		data = pd.read_csv("app/outputs/" + i + ".csv")
		# p = model.predict(Xtest)
		# ytest_df = pd.DataFrame(ytest)
		# p_df = pd.DataFrame(p)

		for i in range(1, len(data.loc[:, '0'].values) + 1):
			index.append(i)

		predicted.append(data.loc[:, '0'].values.tolist())
		indexes.append(index)


		# mean_squared_error(ytest, p)
		# pplt.plot(ytest_df.iloc[:-500, :])
		# pplt.plot(p_df.iloc[-500:, :])
		# pplt.show()
		# stock = bb.Share('Wiki', i)

		# Xtrain, Xtest, ytrain, ytest = split(stock, 'Close', normalize = True)
		# Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
		# Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))
		# rnn = RNN([1, 100, 100, 1]) # number of neurons in each layer
		# rnn.fit(Xtrain, ytrain)

	# print("Stocks list: ", best_stocks)
	# print("Predicted: ", predicted)

	sentiment_categories = []
	publishers_sentiments = []
	percent_overall = []

	for i in best_stocks:
		news = get_news(i + " Stock")
		sentiment_categories.append(news["sentiment_categories"])
		publishers_sentiments.append(news["publishers_sentiments"])
		# print("Publisher sentiments: ", publishers_sentiments)
		percent_overall.append(analyze_tweets(i + " Stock"))

	# best_stocks.append("Amazon")
	# best_stocks.append("JP")

	context = {
		"labels": ["01-03-2019", "02-03-2019", "03-03-2019", "04-03-2019",
				"05-03-2019", "06-03-2019", "07-03-2019", "08-03-2019", "09-03-2019", "10-03-2019"],
		"companies": ["Google", "Amazon", "L&T", "TCS", "JP"],          
		"Google": [86,114,106,106,107,111,133,221,783,2478],
		"Amazon": [282,350,411,502,635,809,947,1402,3700,5267],
		"L&T": [168,170,178,190,203,276,408,547,675,734],
		"TCS": [40,20,10,16,24,38,74,167,508,784],
		"JP": [6,3,2,2,7,26,82,172,312,433],
		"percent_overall": percent_overall,
		"sentiment_categories": sentiment_categories,
		"publishers_sentiments": publishers_sentiments,
		"topic": best_stocks,
		"predicted": predicted,
		"indexes": indexes,
		"name": client.name,
		"client_stocks": client.stocks
	}

		
	return render(request, "stocks.html", context)




def demo_render(request):
	
	# client = Client.objects.filter(name="Sahil Jajodia")[0]


	# print("Budget: ", client.monthly_budget)

	return render(request, "table-layout.html", {})

def get_articles(request, stock):
	
	f = open("app/Articles/" + stock, "r")
	data = f.read()
	
	return JsonResponse(json.dumps({"article": data}), content_type="application/json", safe=False)

def context_qa(request):
	document_seen = 0
	request.session['context_passed'] = 0
	return render(request, "context_qa.html")


def response(request):
	'''
	if request.session['is_asked'] is 0:
		question = request.GET.get('msg')
		document_selected = generate_idf.make_query(question)
		data = {
		'response' : document_selected
		}
		request.session['is_asked'] = 1
		request.session['document_selected'] = document_selected
		return JsonResponse(data)
	else:
		'''
	print(request.session['context_passed'])

	if request.session['context_passed'] is 0:
		context = request.GET.get('msg')
		data = {
			'response': 'Ask your question'
			}
		request.session['context'] = context
		request.session['context_passed'] = 1
		return JsonResponse(data)
	else:
		question = request.GET.get('msg')

		"""
		entity_list = get_named_entities(question)

		for entity in entity_list:
			if search_knowledgebase(entity):

		"""
		readerpath = os.path.join(
				os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
				'fastqa_reader'
			)
		# print(readerpath)
		fastqa_reader = readers.reader_from_file(readerpath)
		#request.session['is_asked'] = 1
		#document_selected = request.GET.get('doc')
		#document_path ='knowledgebase/' + (document_selected.split('/')[-1]).split('.')[0] + '.txt'
		#document_path = 'k'
		'''
		document_path = 'knowledgebase/' + document_selected + '.txt'
		with open(document_path,'r') as myfile:
			support = myfile.read()
			#print (support)
		'''

		context = request.session['context']
		answers = fastqa_reader([QASetting(
		question= question,
		support=[context]
		)])
		print(question, "\n")
		print("Answer: " + answers[0][0].text + "\n")
		data = {
		'response': answers[0][0].text
		}
		return JsonResponse(data)

def wikisearch(request):
	for x in request.session['subjects']:
		wikisearch = wikipedia.search(x)
		search_terms = list(OrderedDict.fromkeys(wikisearch))
		for y in search_terms:
			page = wikipedia.page(y)
			title = unicodedata.normalize('NFKD', page.title)\
				.encode('ascii', 'ignore')
			content = unicodedata.normalize('NFKD', page.content)\
				.encode('ascii', 'ignore')

			# path to knowledge base (downloaded)

			datapath = os.path.join(os.path.dirname(os.path.dirname(sys.path(__file__))),
							"knowledgebase") + title
			with open(datapath, 'w') as datafile:
				print('Writing file: %s\n' % (title))
				datafile.write(content)
	return True
