"""
calculate time to maturity
"""

import datetime as dt
import pandas as pd


holidayPath = "../../universe/holidays/"

class OptionTime():
    def __init__(self, country, holidayPath=holidayPath):
        self.country = country
        self.holidayPath = holidayPath
        self.set_holidays()

    def set_holidays(self):
        self.holidays = getHolidays(self.country, holidayPath=self.holidayPath)
        return self.holidays

    def get_T(self, timeType="days"):

        annualDays = getAnnualTradingDays(dt.datetime.today().year, self.country, holidayPath=self.holidayPath)

        if timeType == "days":
            self.T = len(annualDays)
        elif timeType == "sessions":
            self.T = len(annualDays)*2
        else:
            # calendar
            self.T = 365

        return self.T

    def get_t(self, today, expiration, timeType="days", isMorning=True):

        if timeType == "days":
            self.t = len(getDaysToMaturity(today, expiration, self.holidays))
        else:
            self.t = getNumSessionsToMaturity(today, expiration, self.holidays, isMorning=isMorning)
        return self.t


    def get_ttm(self, today, expiration, timeType="days", isMorning=True):
        """
        :param today: today's date
        :param expiration: expiration date
        :param timeType: days or sessions
        :param isMorning: boolean - morning or after lunch
        :return:
        """

        t = self.get_t(today, expiration, timeType=timeType, isMorning=isMorning)
        T = self.get_T(timeType=timeType)

        return t/T


def convertDate(date):
    """
    converts a string or int date 20180315 to a datetime object
    """
    if isinstance(date, dt.datetime):
        return date
    else:
        if isinstance(date, int):
            date = str(date)
        return dt.datetime(int(date[:4]), int(date[4:6]), int(date[-2:]))

def getAnnualTradingDays(year, country, holidayPath=holidayPath):

    weekmask = "Mon Tue Wed Thu Fri"
    holidays = getHolidays(country, holidayPath=holidayPath)
    dt_holidays = getDatetimeHolidays(holidays)
    days = pd.bdate_range(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31), freq="C",\
                          weekmask=weekmask, holidays=dt_holidays)

    return days

def getHolidays(country, holidayPath=holidayPath):
    """
    country: china, hongkong

    :return: list of datetime objects [datetime(year, month, day)]
    """

    holidays = pd.read_csv(holidayPath + "/%s.csv" %(country), sep=",")
    return holidays.Date.values.astype(str)

def getDatetimeHolidays(holidays):
    return [convertDate(x) for x in holidays]

# get trading days between now and expiration
def getDaysToMaturity(today, expiration, holidays):
    """
    today: int or str date
    expiration: int or str expiration date
    session: morning or afternoon session
    :return:
    """
    weekmask = "Mon Tue Wed Thu Fri"
    dt_holidays = getDatetimeHolidays(holidays)

    # list of days between today and expiration
    return pd.bdate_range(convertDate(today), convertDate(expiration), freq="C", weekmask=weekmask, holidays=dt_holidays)

def getNumSessionsToMaturity(today, expiration, holidays, isMorning=True):

    dtm = getDaysToMaturity(today, expiration, holidays)

    ## ?? subtract 1 for only expiration day
    num_sessions = 2*len(dtm)

    # subtract 1 session if afternoon
    if isMorning:
        return num_sessions
    else:
        return num_sessions - 1

