<div align="center">
<h1 align="center">Chat with Netty</h1>
A customizable LLM specialized in Computer Networking and Cybersecurity.
<br/>
</div>

## How to use

<!-- Setup .env -->
1. Create a `.env` file in the root directory.
```.env
LEPTON_WORKSPACE_TOKEN=YOUR_LEPTON_WORKSPACE_TOKEN
```
2. Set up the environment variables.
```shell
export $(cat .env | xargs)
```
```powershell
cat .env | %{ $_ -replace "^(.*)=(.*)$", "Set-Item Env:$($matches[1]) $matches[2]" } | iex
```
3. Build Next.js App
```shell
cd web && npm install && npm run build
```
4. Set up virtual environment
```shell
cd ..
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
5. Run the server
```shell
python chat_with_netty.py
```

## One-liner brainless setup
```shell
export $(cat .env | xargs) && cd web && npm install && npm run build && cd .. && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python chat_with_netty.py
```
```powershell
cat .env | %{ $_ -replace "^(.*)=(.*)$", "Set-Item Env:$($matches[1]) $matches[2]" } | iex; cd web; npm install; npm run build; cd ..; python -m venv venv; .\venv\Scripts\Activate.ps1; pip install -r requirements.txt; python chat_with_netty.py
```

## Environment Variables
```.env
BACKEND=DUCKDUCKGO
LLM_USE_CUSTOM_SERVER=True
LLM_REMOTE_URL=free.v36.cm
LLM_REMOTE_MODEL=gpt-4o-mini
LLM_REMOTE_API_KEY=[YOUR_API_KEY]
LEPTON_WORKSPACE_ID=[YOUR_LEPTON]
LEPTON_WORKSPACE_TOKEN=[YOUR_LEPTON]
LEPTON_LLM_MODEL=mixtral-8x7b
LEPTON_ENABLE_AUTH_BY_COOKIE=True
RELATED_QUESTIONS=True
KV_NAME=netty-chat.kv

```