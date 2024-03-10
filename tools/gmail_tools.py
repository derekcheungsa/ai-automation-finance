from langchain.tools import tool
import requests


class GmailTools():

  @tool("Send Email")
  def email(full_report):
    """Useful to send an email to the user with the report summary.
    The input to this tool is the text for the full report
    """

    # Initiate a POST request to the send email
    response = requests.post(
        "https://n8n.aifornoncoders.com/webhook/e093a45a-4da8-4180-b176-281775ac96fe",
        json={'input': full_report})
    # Parse the JSON response
    answer = response.text
    return answer
