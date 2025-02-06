

# DO NOT PUT YOUR API KEYS HERE. Put them in `my_platform.sh` and make sure that file is in `.gitignore`.


## Mistral, the preferred model.

#export LLM_API_URL=https://api.mistral.ai/v1
#export LLM_API_KEY=xxx
#export LLM_MODEL=codestral-latest
#export LLM_CONFIG_EXTRA="{}"


## Local model through Jan.ai. Requires lots of RAM. GPU is recommended.

#export LLM_API_URL=http://127.0.0.1:1337/v1
#export LLM_API_KEY=
#export LLM_MODEL=qwen2.5-coder-7b-instruct
#export LLM_CONFIG_EXTRA="{}"


## Groq. Fast but the models generate bad code.

#export LLM_API_URL=https://api.groq.com/openai/v1
#export LLM_API_KEY=xxx
#export LLM_MODEL=gemma2-9b-it
#export LLM_CONFIG_EXTRA="{'prepend_prefix':True}"


## AIML. Free tier has a prompt size limit.

#export LLM_API_URL=https://api.aimlapi.com/v1
#export LLM_API_KEY=xxx
#export LLM_MODEL=mistralai/codestral-2501
#export LLM_CONFIG_EXTRA="{}"


## CodeGPT. The API is similar to OpenAI but not the same.

#export LLM_API_URL=https://api.codegpt.co/api/v1
#export LLM_API_KEY=xxx
#export LLM_MODEL=codestral
#export LLM_CONFIG_EXTRA="{'codegpt':True}"

