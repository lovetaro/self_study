
import pyalgotrade
import fix_yahoo_finance as yf
import pandas_datareader.data as pdr

from pyalgotrade.dataseries.bards import BarDataSeries
ds1= BarDataSeries()
yf.pdr_override()
from pyalgotrade.feed import csvfeed


feed = csvfeed.Feed("Date", "%Y-%m-%d")
feed.addValuesFromCSV('orcl.csv')
for dateTime, value in feed:
    print dateTime, value

from pyalgotrade import dataseries, technical
from pyalgotrade.feed import csvfeed


class Accumulator(technical.EventWindow):
    def getValue(self):
        ret = None
        if self.windowFull():
            ret = self.getValues().sum()
        return ret
seqDS = dataseries.SequenceDataSeries()
accum = technical.EventBasedFilter(seqDS, Accumulator(3))

for i in range(0,50):
    seqDS.append(i)

print(accum[0])
print(accum[2])
print(accum[3])
print(accum[-1])



from pyalgotrade.technical import ma
from pyalgotrade.feed import BaseFeed

sdi = csvfeed.Feed("Date", "%Y-%m-%d")
sdi.addValuesFromCSV("sdi.csv")

sdi_price = dataseries.SequenceDataSeries()
for value in sdi['Adj Close']:
    sdi_price.append(value)


from pyalgotrade import eventprofiler
from pyalgotrade.technical import stats, roc, ma


class BuyOnGap(eventprofiler.Predicate):   #base/parent
    def __init__(self, feed):
        super(BuyOnGap, self).__init__()    #same as super().__init__()
        
        stdDevPeriod = 20
        smaPeriod = 20
        self.__returns = {}
        self.__stdDev = {}
        self.__ma = {}
        
        for instrument in feed.getRegisteredInsturments():
            priceDS = feed[instrument].getAdjCloseChange(priceDS, 1)
            self.__returns[instrument] = roc.RateOfChange(priceDS, 1)
            self.__stdDev[instrument] = stats.StdDev(self.__returns[instrument], stdDevPeriod)
            self.__ma[insturment] = ma.SMA(priceDS, smaPeriod)
    
    def __gappedDown(self, instument, bards):
        ret = False  #default
        
        if self.__stdDev[instrument][-1] is not None:
            prevBar = bards[-2]
            currBar = bards[-1]
            low2OpenRet = (currBar.getOpen(True) - prevBar.getLow(Ture)) / float(prevBar(True))
            if low2OpenRet < self.__returns[instrument] - self.__stdDev[instrument]:
                ret = True
    
    def __aboveSMA(self, instrument, bards):
        ret = False
        if self.__ma[instrument][-1] is not None and bards[-1].getOpen(True) > self.__ma[instrument][-1]:
            ret = True
        return ret

    def eventOccurred(self, instrument, bards):
        ret = False
        if self.__gappedDown(instrument, bards) and self.__aboveSMA(instrument, bards):
            ret = True
        return ret


def main(plot):
    instruments = ["AA", "AES", "AIG"]
    feed = csvfeed.Feed("Date", "%Y-%m-%d")
    for i in instruments:
        temp = pdr.get_data_yahoo(i, '2013-01-01', '2018-03-28')
        temp = temp['Adj Close']
        temp.to_csv('temp.csv')
        feed.addValuesFromCSV("temp.csv")
        
    predicate = BuyOnGap(feed)
    eventProfiler = eventprofiler.Profiler(predicate, 5, 5)
    eventProfiler.run(feed, True)

    results = eventProfiler.getResults()
    print "%d events found" % (results.getEventCount())
    if plot:
        eventprofiler.plot(results)                                                   

