# AI Agents Crash Course: Build with Python & OpenAI

## Environment Setup

1) Sign up to OpenAI and [generate an OpenAI API Key](https://platform.openai.com/api-keys)
2) Copy the contents of [.env.template](.env.template) to a file called `.env` and add your api key to to it. You can use the UI or execute these commands in the terminal:
```
cp .env.template .env
code .env
# Add in your OpenAI API Key
```
3) Open our first notebook, [notebooks/simplest_agent.ipynb) and execute the first cell

## Our First Agent

* [Understanding async/await in Python](https://realpython.com/async-io-python/)
* [OpenAI Tracing UI](https://platform.openai.com/logs?api=traces)

## MCP
* [Smithery.ai MCP Hub](https://smithery.ai/)
* [Exa Search](https://exa.ai/)
* [OpenAI Tools Documentation](https://openai.github.io/openai-agents-python/tools/) for the exercise

## Running the Chatbot
One-time setup
```
cd chatbot && chainlit run 4_authentication.py --port 10000 --host 0.0.0.0
```

The URL for our demo logo: 
```
https://upload.wikimedia.org/wikipedia/commons/e/e3/Udemy_logo.svg
```

### Adding Authentication
1) Generate a Chainlit Secret: `chainlit create-secret`
2) Add the secret to `.env`
3) Pick an arbitrary username and password and fill in the `CHAINLIT_USERNAME` and `CHAINLIT_PASSWORD` values in `.env`

## Deployment
Here is the command:
```
chainlit run chatbot/4_authentication.py --port 10000 --host 0.0.0.0
```

## AgentBuilder

* [The AgentBuilder UI](https://platform.openai.com/agent-builder)

### Datasets for the File Search Tool
* [calorie_database.txt](https://nutrition-datasets.s3.amazonaws.com/calorie_database.txt)
* [questions_output.txt](https://nutrition-datasets.s3.amazonaws.com/questions_output.txt)

### RAG & File Search Instructions
```
You are a helpful nutrition assistant giving out calorie information. You give concise answers.
If you are asked about calorie information, always read the data from the Calorie Database even if you know the calorie by heart as the Calorie Database has the most up-to-date information.
```

### MCP in AgentBuilder
Exa MCP URL:
```
https://mcp.exa.ai/mcp?exaApiKey=<<YOUR API KEY>>
```

Ingredient Agent Instructions:
```
You are a helpful nutrition assistant breaking down meals into ingredients. You give concise answers.

*Always use the `web_search_exa` tool to search the web for the ingredients for the meal before you search for calories in the Calorie Database. Use this tool even if you know the ingredients as the tool might have more up-to-date information.

* Give the name and the size for each ingredients for a two-person serving

* Your only job is to firgure out the ingredients. Subsequent agents will take care of the calorie calculation.

This is the response format you want to use (markdown). Don't add anything else to the response:
```
Meal name: {{meal name}}

Ingredient list for two servings:
* {{ingredient name}}: {{ingredient size}}
```

Calorie Agent Instructions
```
You are a calorie lookup agent that takes ingredients on it's input, looks up calorie information using the Calorie Database tool and responds with calorie information in a format provided below. You don't need to add references to results.

You strictly follow these steps:
1) You  always read the calorie information for the ingredients using the `Calorie Database` File Search tool, even if you know the calorie by heart. The `Calorie Database` File Search tool has the most up-to-date information. Once 
2) Once you have the calorie for each ingredients, you add them up to calculate the total calories.
3) Then you fill in every part of the template below and respond.

You never ask followup questions but do whatever is needed to fill in the template below. 
You always fill in the full list of ingredients.

This is how your response must look (md format):
```
# {{meal name}}
_Serving size: {{serving size}}

_Total calories: {{total_calories}}_ calories

Ingredients: 
* {{Ingredient 1}} ({{ingredient size}}): {{ calories }} calories

```

User Prompt:
```
How many calories in a full english breakfast?
```

### Control Structures and Structured Messages
Instructions for the Off-Topic Agent:
```
if the topic is about calories, set the output's `topic` to `calories`, `other` otherwise
```

Instructions for the "Error Message" Agent:
```
Politely inform the user that this workflow is focused exclusively on nutrition topics and cannot assist with non-nutrition-related inquiries.
```

## Chatkit
* [The OpenAI ChatKit Starter App](https://github.com/openai/openai-chatkit-starter-app)

# Datasets
Here are the originals of the datasets we use in the course:
* [Calorie Database](https://www.kaggle.com/datasets/kkhandekar/calories-in-food-items-per-100-grams)
* [Nutritionist Q&A](https://huggingface.co/datasets/RaniRahbani/nutritionist)
