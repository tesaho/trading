"""
base class for options
"""

from helpers.instruments import Instrument

class option_class():
    def __init__(self, product, exchange):
        self.product = product
        self.exchange = exchange
        self._getInstruments()

    def _getInstruments(self):
        inst = Instrument()
        self.optInstr = inst.getInstrument(self.product)
        self.spotInstr = inst.getInstrument(self.optInstr.params.Underlying_Contract)
        self.optTickIncr = self.optInstr.params.Tick_Size
        self.optTickValue = self.optInstr.params.Tick_Value
        self.spotTickIncr = self.spotInstr.params.Tick_Size
        self.spotTickValue = self.spotInstr.params.Tick_Value

    def __str__(self):
        return self.product + " " + self.exchange


class Option(option_class):
    def __init__(self, product, exchange):
        super().__init__(product, exchange)
        self.exchange_symbol = None
        self.optionName = None
        self.optType = None
        self.strike = None
        self.term = None
        self.params = None
        self.maturity = None
        self.priceRoundLevel = None
        self.riskAdjTv = None
        self.tv = None
        self.buyCalcEdge = None
        self.sellCalcEdge = None

    def setInstrument(self, optionName, optType, strike, term, exchange_symbol=None ):
        """
        optionName: "C1806M_2600"

        sets option name class ticker
        """
        self.exchange_symbol = exchange_symbol
        self.optionName = optionName
        self.optType = optType
        self.strike = strike
        self.term = term
        self.params = self.optInstr.params
        self.maturity = self.params.Maturity[term]
        self.priceRoundLevel = len(self.optTickIncr.split(".")[-1])

        return self

    def setOptionFields(self, fieldDict):

        self.__dict__.update(fieldDict)