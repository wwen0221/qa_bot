# RAG-based Q&A bot

## Step 1
Create python virtual environment and activate:

`python -m venv new-venv`

`source new-venv/bin/activate`

## Step 2
Install uvicorn package
`pip install uvicorn`

## Step 3
Clone the repo

## Step 4 
Navigate to the project directory and run:
`pip install -r requirements.txt`

## Step 5
Create .env file that consists of:
```
OPENAI_KEY = 'YOUR_OPENAI_KEY'
TAVILY_KEY = 'YOUR_TAVILY_KEY'
```
## Step 6
Start uvicorn by running:

```uvicorn app:app --reload --port 8000```

------------
# To call the api:

## You could use postman to call the bot: 
http://localhost:8000/query?question=how much is iphone 15 pro?

![image](https://github.com/user-attachments/assets/d0b08aed-f5d3-4bf1-aa49-d4e517ef6eee)

## If you wish to use python script:
```
import requests
url = "http://localhost:8000/query"
params = {"question": "Tell me about the iPhone 15 Pro"}
response = requests.get(url, params=params)
print(response.json())
```

## If you wish to use terminal:
`curl "http://localhost:8000/query?question=Tell%20me%20about%20the%20iphone%2015%20pro"`



