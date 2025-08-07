# app/modelsarch.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union


class PriceChartDataPointAPI(BaseModel):
    date: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[Union[int, float]] = None
    dma50: Optional[float] = None
    dma200: Optional[float] = None

class PriceChartResponseAPI(BaseModel):
    priceData: List[PriceChartDataPointAPI]

class PriceChange(BaseModel):
    absolute: str
    percent: str
    isPositive: bool

class ChartDataItem(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[int] = None
    sales: Optional[float] = None
    netProfit: Optional[float] = None
    eps: Optional[float] = None
    cashFromOperating: Optional[float] = None
    netCashFlow: Optional[float] = None
    cmp: Optional[float] = None
    pe: Optional[float] = None
    value: Optional[float] = None
    Promoters: Optional[float] = None
    FII: Optional[float] = None
    DII: Optional[float] = None
    Public: Optional[float] = None

class PeerChart(BaseModel):
    data: List[ChartDataItem]
    title: str
    dataKey: str

# New model for individual indicator signals
class IndicatorSignal(BaseModel):
    name: str  # e.g., "RSI", "MACD"
    value: Optional[Union[float, str, Dict[str, Optional[float]]]] = None # Raw value(s) of indicator
    decision: str  # "Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"

class StockQuote(BaseModel): # For the stock list
    id: str
    name: str
    symbol: str
    currentPrice: Union[float, str]
    marketCapFormatted: str
    # New fields for technical indicators:
    signals: Optional[List[IndicatorSignal]] = None
    overallSignal: Optional[str] = None
    # For debugging, can be removed later
    debug_score: Optional[float] = None


class StockDetailResponse(BaseModel):
    id: str
    name: str
    symbol: str
    bseCode: Optional[str] = None
    nseCode: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    marketCapFormatted: str
    currentPrice: Union[float, str]
    priceChange: PriceChange
    yearHighLow: str
    dayHighLow: str
    stockPE: Union[float, str]
    bookValueFormatted: str
    dividendYieldFormatted: str
    roceFormatted: str
    roeFormatted: str
    faceValue: Union[int, str]
    rsiValue: Union[float, str] # This is from calculations.py, single value
    aboutInfo: str
    keyPoints: List[str]
    pros: List[str]
    cons: List[str]
    priceVolumeChartData: List[ChartDataItem]
    peerCmpChart: PeerChart
    peerPeChart: PeerChart
    quarterlyFinancialsChartData: List[ChartDataItem]
    quarterlyEPSChartData: List[ChartDataItem]
    annualFinancialsChartData: List[ChartDataItem]
    cashFlowsChartData: List[ChartDataItem]
    shareholdingPieData: List[ChartDataItem]
    shareholdingTrendData: List[ChartDataItem]
    peerComparison: List[Dict[str, Any]]
    quarterlyResults: List[Dict[str, Any]]
    profitAndLoss: List[Dict[str, Any]]
    balanceSheet: List[Dict[str, Any]] 
    balanceSheetLiabilitiesChartData: Optional[List[Dict[str, Any]]] = None # New
    balanceSheetAssetsChartData: Optional[List[Dict[str, Any]]] = None    # New
    cashFlows: List[Dict[str, Any]]
    shareholdingHistory: List[Dict[str, Any]]

    # For Ratios Section
    ratiosTableData: Optional[List[Dict[str, Any]]] = None
    efficiencyDaysChartData: Optional[List[Dict[str, Any]]] = None # For Debtor/Inv/Payable Days
    roceTrendChartData: Optional[List[Dict[str, Any]]] = None    # For ROCE trend

    # Add fields for growth metrics
    compoundedSalesGrowth: Optional[Dict[str, float]] = None # e.g., {"10 Years": 5.0, "5 Years": 3.0}
    compoundedProfitGrowth: Optional[Dict[str, float]] = None
    stockPriceCagr: Optional[Dict[str, float]] = None
    returnOnEquityTrend: Optional[Dict[str, float]] = None # Or however you structure ROE trend data