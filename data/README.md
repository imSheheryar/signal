# Data

The MAEC corpus is not redistributed here. Download it from the upstream repository:

```bash
git clone --depth 1 https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction.git data/maec
```

MAEC is licensed CC BY-SA 4.0. Prices are fetched with yfinance and are not redistributed.

Expected layout:

```
data/maec/MAEC_Dataset/<YYYYMMDD>_<TICKER>/{text.txt,features.csv}
```
