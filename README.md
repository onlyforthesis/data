
# 📈 交易策略最佳化技術  
### **基於利潤-價格分佈與遺傳演算法**  
**Thesis Project: Profit-Price Distribution Based Trading Strategy Optimization with Genetic Algorithms**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Last Commit](https://img.shields.io/github/last-commit/onlyforthesis/114-)  

---

## 📌 **專案簡介 (Introduction)**  
本專案提出並驗證一套 **以利潤-價格分佈 (Profit-Price Distribution) 為核心的交易策略最佳化框架**，結合 **PPTS (Profit-Price Trading Strategy)** 與 **GAPPTS (Genetic Algorithm-based PPTS)**，用於優化多目標績效指標（**收益率**、**Calmar 比率**）。  

研究以 **台積電 (2330.TW)** 為例，透過歷史數據進行 **訓練 (2019-2023)** 與 **回測 (2024)**，並與基線策略及 AI/ML 預測方法進行比較。  

**English Abstract**  
This project implements and evaluates a trading strategy optimization framework based on **Profit-Price Distribution** integrated with **Genetic Algorithm (GA)**. The proposed approach, **PPTS + GAPPTS**, aims to optimize multi-objective performance metrics such as **return** and **Calmar ratio**. Empirical tests are conducted using **TSMC (2330.TW)** historical data, with benchmark comparisons against standard strategies.

---

## 🔍 **專案特色 (Key Features)**  
✔ **PPTS 策略構建**  
　將價格區間化並計算各區間的平均利潤與獲利機率，產生初步買賣訊號。  

✔ **GAPPTS 參數優化**  
　利用 **遺傳演算法** 搜尋最適化參數組合：  
　- **n_bins**：價格分區數  
　- **window**：滾動視窗長度  
　- **profit_threshold**：利潤門檻  

✔ **多目標最佳化 (Return & Calmar Ratio)**  
　雙目標設計確保策略兼顧 **獲利能力** 與 **風險控制**。  

✔ **完整回測框架**  
　- **訓練資料**：2019-2023  
　- **測試資料**：2024  
　- **基準策略比較**：Buy & Hold、ML 模型 (LSTM, RF, XGBoost)  

✔ **圖表 & 報表輸出**  
　支援 **績效分析圖表**、**遺傳演算法收斂曲線**、**論文表格對應輸出 (Table 3.13~3.18)**。  

---

## 🏗 **系統架構 (Architecture)**  
```
┌─────────────────────────┐
│     Data Preprocessing  │
│  (TSMC Historical Data) │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│  PPTS Strategy Builder  │
│ (Profit-Price Analysis) │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│ GA-based Parameter Tuning│
│   (n_bins, window, ... ) │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│   Backtesting & Metrics │
│ (Return, Calmar, Sharpe)│
└─────────────────────────┘
```

### **方法一：Profit_Price_based_Trading_Strategy_Construction_Algorithm_(PPTS) .ipynb**

論文3-3虛擬碼
Profit-Price based Trading Strategy Construction Algorithm (PPTS)實際程式

### **20241203基因演算法.ipynb**

論文3-3虛擬碼
GA-based Profit-Price based Trading Strategy Optimization Algorithm (GAPPTS)實際程式

### **update_stock.ipynb**

4.4 不同行業股票預測效能比較(塞入不同股票代號)


### **演算法表現比較**
4.5 與不同方法預測股票成效比較分析
