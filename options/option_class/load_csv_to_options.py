"""
take a greek csv and transfer to option_term and option_class

"""

import pandas as pd
from options.option_class.option_term import OptionTerm


## take csv
def getOptionCsv(csvPath):

    # set index to optNames
    df = pd.read_csv(csvPath, index_col=0)
    df.loc[:, "optType"] = [x[0] for x in df.index]
    df.loc[:, "mid"] = (df.optBid + df.optAsk) / 2.0
    df.loc[:, "term"] = [x[1:5] for x in df.index]
    df.sort_index(ascending=True, inplace=True)

    return df


## create optionTerm
def getOptionTerm(product, exchange, optNames, exchange_symbol=None):

    optTerm = OptionTerm(product, exchange)
    optTerm.setTerm(optNames, exchange_symbol=exchange_symbol)

    return optTerm

def getOptions(df, product, exchange, exchange_symbol=None):

    optNames = df.index.values
    # create option term
    optTerm = getOptionTerm(product, exchange, optNames, exchange_symbol=exchange_symbol)
    # update option term with csv data
    optTerm = setOptions(df, optTerm)

    return optTerm

def setOptions(df, optTerm):

    df_dict = df.to_dict(orient="index")

    for optionName in optTerm.options.keys():
        option = optTerm.options[optionName]
        fieldDict = df_dict[optionName]
        fieldDict["optionName"] = optionName
        option.setOptionFields(fieldDict)

    return optTerm


