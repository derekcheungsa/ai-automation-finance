from langchain.tools import tool
import requests

from openbb import obb
import pandas as pd


class OpenBBTools():

  @tool("Useful to get income statement for a company")
  def income(ticker):
    """Useful to getting income statement for a company.
    The input to this tool should be the ticker symbol of the company to get income statement for.   For example, NVDA.
    """

    df = pd.DataFrame()

    df = (obb.equity.fundamental.income(ticker,
                                        provider="yfinance",
                                        limit=1,
                                        period="annual").to_df())

    answer = df.to_json()
    return answer
