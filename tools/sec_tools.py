from langchain.tools import tool
import requests


class SecTools():

  @tool("Ask question about from SEC 10-K about NVDIA")
  def sec(question):
    """Useful to answering fundamental questions about NVDIA from SEC 10-K
    The input to this tool should be the question to be answered. For example, what does Nvdia do to earn money?
    """

    # Initiate a POST request to the SEC API
    response = requests.post(
        "https://n8n.aifornoncoders.com/webhook/d707d430-f73a-4378-a6da-b7427c704cf9",
        json={'input': question})
    # Parse the JSON response
    answer = response.text
    return answer
