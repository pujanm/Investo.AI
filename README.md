# SARAKR.AI
We built this WebApp in CodeShastra 5.0 - Round 0.

Everyone has the excitement of knowing which party will be ruling India from 2019 onwards. So we came up with this web app wherein you can type a trending political topic for say Rafale Deal or Demonetization or any topic you are interested in, you get deep insights on other people's perspective on these topics from various news articles, blogs, and Twitter. By having a look over all the analysis we hope that you will be able to make a decision on choosing your party. We didn't predict which party will win as then our application will be considered biased and now our application says what people say. We also have provided you with a bot where you can feed it some context and then asked questions based on that.

### Here is how the WebApp looks:

### Dashboard where you can search the topic you want analysis on:
![screenshot from 2019-02-11 16-18-21](https://user-images.githubusercontent.com/26873907/52565545-1f986d80-2e2d-11e9-9d23-95497956641d.png)
![screenshot from 2019-02-11 16-18-47](https://user-images.githubusercontent.com/26873907/52565563-28893f00-2e2d-11e9-9813-d36824351c94.png)
![screenshot from 2019-02-11 18-50-39](https://user-images.githubusercontent.com/26873907/52566110-e103b280-2e2e-11e9-8e7e-cb5a409e2936.png)
![screenshot from 2019-02-11 16-19-27](https://user-images.githubusercontent.com/26873907/52565562-27f0a880-2e2d-11e9-9b9d-a912b7da4fd4.png)

### General political trends:
![screenshot from 2019-02-11 16-19-47](https://user-images.githubusercontent.com/26873907/52566150-ff69ae00-2e2e-11e9-8acb-34994f9adb48.png)
![screenshot from 2019-02-11 16-20-05](https://user-images.githubusercontent.com/26873907/52566149-ff69ae00-2e2e-11e9-8cd9-8a5b706c965b.png)

### Context Based Question & Answering:
![screenshot from 2019-02-11 16-20-38](https://user-images.githubusercontent.com/26873907/52566214-317b1000-2e2f-11e9-984e-3872047fec0c.png)
![screenshot from 2019-02-11 16-21-29](https://user-images.githubusercontent.com/26873907/52566213-317b1000-2e2f-11e9-8768-bd4794780d9b.png)

### SETUP:
#### Basic System Prerequisites:
```
Python == 3.6
virtualenv >= 16.0.0
Operating System == Linux || MacOS
```
#### Clone this repository:
```
git clone https://github.com/pujanm/SARKAR.AI.git
cd SARKAR.AI/
```
#### Data Setup:
```
wget -O fast.zip https://www.dropbox.com/sh/vnmc9pq4yrgl1sr/AACkQjvkuXXGjSRzJQovDrz-a/qa/squad/fastqa.zip?dl=0
unzip fastqa.zip && mv fastqa fastqa_reader
./download.sh
```
#### WebApp Setup:
```
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en
```
