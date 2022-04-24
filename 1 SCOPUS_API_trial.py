from pybliometrics.scopus import ScopusSearch
import pandas as pd


s = ScopusSearch(
    ' ',
    download=False,
    subscriber=False)

print(s)


# df = pd.DataFrame(pd.DataFrame(s.results))
# print(df.columns)
