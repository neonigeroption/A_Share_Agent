import akshare as ak
try:
    df = ak.stock_zh_a_daily(symbol="sz000938", start_date="20240101", end_date="20240110")
    print("SINA SUCCESS")
    print(df.head(2))
except Exception as e:
    print("SINA FAILED")
    print(e)
