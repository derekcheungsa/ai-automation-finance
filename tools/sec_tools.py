from langchain.tools import tool
import requests
import os

N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL')
FLOWISE_NVDA_URL = os.getenv('FLOWISE_NVDA_URL')
FLOWISE_AMD_URL = os.getenv('FLOWISE_AMD_URL')


class SecTools():

  @tool("Ask question about from SEC 10-K about NVDIA")
  def sec_nvda(question):
    """Useful to answering fundamental questions about NVDIA from SEC 10-K
    The input to this tool should be the question to be answered. For example, what does Nvdia do to earn money?
    """

    # Initiate a POST request to the SEC API
    # N8N
    #
    # response = requests.post(N8N_WEBHOOK_URL, json={'input': question})
    # Parse the JSON response
    # answer = response.text

    # FLOWISE
    response = requests.post(N8N_WEBHOOK_URL, json={'question': question})
    answer = response.json().get('text', 'No text available')

    return answer

  @tool("Ask question about from SEC 10-K about AMD")
  def sec_amd(question):
    """Useful to answering fundamental questions about AMD from SEC 10-K
    The input to this tool should be the question to be answered. For example, what does AMD do to earn money?
    """

    # Initiate a POST request to the SEC API
    # N8N
    #
    # response = requests.post(N8N_WEBHOOK_URL, json={'input': question})
    # Parse the JSON response
    # answer = response.text

    # FLOWISE
    response = requests.post(FLOWISE_AMD_URL, json={'question': question})
    answer = response.json().get('text', 'No text available')

    return answer
