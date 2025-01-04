<div align="center">
<h1 align="center">Chat with Netty</h1>
Build your own conversational search engine using less than 500 lines of code.
<br/>
<a href="https://search.lepton.run/" target="_blank"> Live Demo </a>
<br/>
<img width="70%" src="https://github.com/leptonai/search_with_lepton/assets/1506722/845d7057-02cd-404e-bbc7-60f4bae89680">
</div>


## Features
- Built-in support for LLM
- Built-in support for search engine
- Customizable pretty UI interface
- Shareable, cached search results

## Setup Search Engine API
There are two default supported search engines: Bing and Google.
 
### Bing Search
To use the Bing Web Search API, please visit [this link](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api) to obtain your Bing subscription key.

### Google Search
You have three options for Google Search: you can use the [SearchApi Google Search API](https://www.searchapi.io/) from SearchApi, [Serper Google Search API](https://www.serper.dev) from Serper, or opt for the [Programmable Search Engine](https://developers.google.com/custom-search) provided by Google.

## Setup LLM and KV

> [!NOTE]
> We recommend using the built-in llm and kv functions with Lepton. 
> Running the following commands to set up them automatically.

```shell
pip install -U leptonai openai && lep login
```

## Obtain Your Lepton AI Workspace Token
You can copy your workspace toke from the Lepton AI Dashboard &rarr; Settings &rarr; Tokens.


## Build

1. Set Bing subscription key
```shell
export BING_SEARCH_V7_SUBSCRIPTION_KEY=YOUR_BING_SUBSCRIPTION_KEY
```
2. Set Lepton AI workspace token
```shell
export LEPTON_WORKSPACE_TOKEN=YOUR_LEPTON_WORKSPACE_TOKEN
```
3. Build web
```shell
cd web && npm install && npm run build
```
4. Run server
```shell
BACKEND=BING python search_with_lepton.py
```

For Google Search using SearchApi:
```shell
export SEARCHAPI_API_KEY=YOUR_SEARCHAPI_API_KEY
BACKEND=SEARCHAPI python search_with_lepton.py
```

For Google Search using Serper:
```shell
export SERPER_SEARCH_API_KEY=YOUR_SERPER_API_KEY
BACKEND=SERPER python search_with_lepton.py
```

For Google Search using Programmable Search Engine:
```shell
export GOOGLE_SEARCH_API_KEY=YOUR_GOOGLE_SEARCH_API_KEY
export GOOGLE_SEARCH_CX=YOUR_GOOGLE_SEARCH_ENGINE_ID
BACKEND=GOOGLE python search_with_lepton.py
```


## Deploy

You can deploy this to Lepton AI with one click:

[![Deploy with Lepton AI](https://github.com/leptonai/search_with_lepton/assets/1506722/bbd40afa-69ee-4acb-8974-d060880a183a)](https://dashboard.lepton.ai/workspace-redirect/explore/detail/search-by-lepton)

You can also deploy your own version via

```shell
lep photon run -n search-with-lepton-modified -m search_with_lepton.py --env BACKEND=BING --env BING_SEARCH_V7_SUBSCRIPTION_KEY=YOUR_BING_SUBSCRIPTION_KEY
```

Learn more about `lep photon` [here](https://www.lepton.ai/docs).


========================================================================================================================================================================================================================================

# Lepton Search
Build your own conversational search engine using less than 500 lines of code.

See a live demo site https://search.lepton.run/

The source code of this project lives [here](https://github.com/leptonai/search_with_lepton/). This README will detail how to set up and deploy this project on Lepton's platform.

## Setup Search Engine API

You have a few options for setting up your search engine API. You can use Bing or Google, or if you just want to very quickly try the demo out, use the lepton demo API directly.

### Bing

If you are using Bing, you can subscribe to the bing search api [here](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api). After that, write down the Bing search api subscription key. We follow the convention and name it `BING_SEARCH_V7_SUBSCRIPTION_KEY`. We recommend you store the key as a secret in Lepton.

### Google

If you choose to use Google, you can follow the instructions [here](https://developers.google.com/custom-search/v1/overview) to get your Google search api key. We follow the convention and name it `GOOGLE_SEARCH_API_KEY`. We recommend you store the key as a secret in Lepton. You will also get a search engine CX id, which you will need as well.

### SearchApi

If you want to use SearchApi, a 3rd party Google Search API, you can retrieve the API key by registering [here](https://www.searchapi.io/). We follow the convention and name it `SEARCHAPI_API_KEY`. We recommend you store the key as a secret in Lepton.

### Lepton Demo API

If you choose to use the lepton demo api, you don't need to do anything - your workspace credential will give you access to the demo api. Note that this does incur an API call cost.


## Deployment Configurations

Here are the configurations you can set for your deployment:
* Name: The name of your deployment, like "my-search"
* Resource Shape: most of heavy lifting will be done by the LLM server and the search engine API, so you can choose a small resource shape. `cpu.small` is usually good enough.

Then, set the following environmental variables.

* `BACKEND`: the search backend to use. If you don't have bing or google set up, simply use `LEPTON` to try the demo. Otherwise, do `BING`, `GOOGLE` or `SEARCHAPI`.
* `LLM_MODEL`: the LLM model to run. We recommend using `mixtral-8x7b`, but if you want to experiment other models, you can try the ones hosted on LeptonAI, for example, `llama2-70b`, `llama2-13b`, `llama2-7b`. Note that small models won't work that well.
* `KV_NAME`: the Lepton KV to use to store the search results. You can use the default `search-with-lepton`.
* `RELATED_QUESTIONS`: whether to generate related questions. If you set this to `true`, the search engine will generate related questions for you. Otherwise, it will not.
* `GOOGLE_SEARCH_CX`: if you are using google, specify the search cx. Otherwise, leave it empty.
* `LEPTON_ENABLE_AUTH_BY_COOKIE`: this is to allow web UI access to the deployment. Set it to `true`.

In addition, you will need to set the following secrets:
* `LEPTON_WORKSPACE_TOKEN`: this is required to call Lepton's LLM and KV apis. You can find your workspace token at [Settings](https://dashboard.lepton.ai/workspace-redirect/settings).
* `BING_SEARCH_V7_SUBSCRIPTION_KEY`: if you are using Bing, you need to specify the subscription key. Otherwise it is not needed.
* `GOOGLE_SEARCH_API_KEY`: if you are using Google, you need to specify the search api key. Note that you should also specify the cx in the env. If you are not using Google, it is not needed.
* `SEARCHAPI_API_KEY`: if you are using SearchApi, a 3rd party Google Search API, you need to specify the api key.

Once these fields are set, click `Deploy` button at the bottom of the page to create the deployment. You can see the deployment has now been created under [Deployments](https://dashboard.lepton.ai/workspace-redirect/deployments). Click on the deployment name to check the details. You’ll be able to see the deployment URL and status on this page.

Once the status is turned into `Ready`, click the URL on the deployment card to access it. Enjoy!
