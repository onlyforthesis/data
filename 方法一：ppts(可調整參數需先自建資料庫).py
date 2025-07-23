import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import gradio as gr

# === 解決 matplotlib 中文顯示問題 ===
import matplotlib
# 設定中文字體，按優先順序嘗試
chinese_fonts = [
    'Microsoft JhengHei',     # Windows 繁體中文
    'Microsoft YaHei',        # Windows 簡體中文
    'SimHei',                 # Windows 黑體
    'Heiti TC',               # macOS 繁體中文
    'Arial Unicode MS',       # 通用 Unicode 字體
    'DejaVu Sans',           # Linux 常見字體
    'WenQuanYi Micro Hei',   # Linux 中文字體
    'sans-serif'             # 後備字體
]

# 設定字體參數
plt.rcParams['font.sans-serif'] = chinese_fonts
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.size'] = 10  # 設定預設字體大小

# 檢查並設定字體
try:
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts[:-1]:  # 排除 sans-serif
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"使用字體: {font}")
            break
    else:
        print("未找到中文字體，使用預設字體")
except Exception as e:
    print(f"字體設定錯誤: {e}")
    # 使用備用設定
    matplotlib.rcParams['font.family'] = ['sans-serif']

# 修正問題1：改進字體設定，徹底解決中文顯示問題
import warnings
import matplotlib.font_manager as fm
import platform
import os

# 隱藏matplotlib中文顯示警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 根據作業系統找尋合適的中文字體
def setup_matplotlib_chinese_fonts():
    system = platform.system()
    
    font_found = False
    font_paths = []
    
    # 根據不同作業系統找尋字體檔案路徑
    if system == 'Windows':
        font_paths = [
            r'C:\Windows\Fonts\msjh.ttc',      # 微軟正黑體
            r'C:\Windows\Fonts\mingliu.ttc',   # 細明體
            r'C:\Windows\Fonts\kaiu.ttf',      # 標楷體
            r'C:\Windows\Fonts\simhei.ttf'     # 簡體黑體
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Microsoft/PMingLiU.ttf',
            '/Library/Fonts/Microsoft/SimHei.ttf'
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/arphic/uming.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        ]
        
    # 檢查字體檔案是否存在，並加入至matplotlib字體列表
    for path in font_paths:
        if os.path.exists(path):
            try:
                prop = fm.FontProperties(fname=path)
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [prop.get_name(), *plt.rcParams['font.sans-serif']]
                print(f"成功載入中文字體: {path}")
                font_found = True
                break
            except Exception as e:
                print(f"字體載入失敗: {path}, 錯誤: {e}")
    
    # 設定備用字體
    if not font_found:
        # 使用標準字體設定
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    return font_found

# 調用字體設定函數
setup_matplotlib_chinese_fonts()

# === 全域參數設定 ===
TICKER = "2330.TW"           # 股票代碼
M = 5                        # 區間數量
HOLD_DAYS = 10               # 持有天數
ALPHA = 0.3                  # 買進訊號門檻（降低門檻）
TARGET_PROFIT_RATIO = 0.5    # 目標利潤比例（降低門檻）
TRANSACTION_COST = 0.001     # 交易成本

# === 資料庫連線設定 ===
def get_db_connection():
    return pyodbc.connect(
        'Driver={SQL Server};'
        'Server=DESKTOP-TOB09L9;'
        'Database=StockDB;'
        'Trusted_Connection=yes;'  # Windows 驗證
    )

# === 從資料庫獲取可用股票代碼 ===
def get_available_tickers():
    """從資料庫獲取所有資料表"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 獲取 StockDB 中所有的資料表
        cursor.execute("""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        AND TABLE_CATALOG = 'StockDB'
        ORDER BY TABLE_NAME
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not tables:  # 如果沒有資料，使用預設值
            return ["StockPrices"]
        return tables
        
    except Exception as e:
        print(f"讀取資料表時發生錯誤: {str(e)}")
        return ["StockPrices"]  # 發生錯誤時返回預設值

def get_default_ticker():
    """獲取預設資料表名稱"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT TOP 1 TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        AND TABLE_CATALOG = 'StockDB'
        ORDER BY TABLE_NAME
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else "StockPrices"
    except Exception as e:
        print(f"讀取預設資料表時發生錯誤: {str(e)}")
        return "StockPrices"

# === 取得股票資料 ===
def get_stock_data(table_name, period="5y"):
    """從指定資料表獲取股票資料"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 動態查詢欄位名稱，避免 SQL 關鍵字衝突
        cursor.execute(f"""
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        """)
        columns = [row[0] for row in cursor.fetchall()]
        # 預設欄位對應
        col_map = {
            'Date': None, 'Open': None, 'High': None, 'Low': None, 'Close': None, 'Volume': None
        }
        # 嘗試自動對應欄位
        for col in columns:
            col_clean = col.lstrip('\ufeff').lower()  # 處理 BOM 問題
            if col_clean in ['date', 'tradedate']:
                col_map['Date'] = col
            elif col_clean in ['open', 'openprice', 'open_price']:
                col_map['Open'] = col
            elif col_clean in ['high', 'highprice', 'high_price']:
                col_map['High'] = col
            elif col_clean in ['low', 'lowprice', 'low_price']:
                col_map['Low'] = col
            elif col_clean in ['close', 'closeprice', 'close_price']:
                col_map['Close'] = col
            elif col_clean in ['volume', 'vol']:
                col_map['Volume'] = col

        # 顯示目前資料表所有欄位，協助使用者檢查
        if not all(col_map.values()):
            print(f"資料表 {table_name} 欄位不足，請確認欄位名稱。")
            print(f"該表實際欄位: {columns}")
            print(f"需包含欄位（不分大小寫）：date/tradedate, open/openprice, high/highprice, low/lowprice, close/closeprice, volume")
            return pd.DataFrame()

        # 查詢資料
        # 避免 SQL 關鍵字衝突，所有欄位名稱都加上中括號
        query = f"""
        SELECT 
            [{col_map['Date']}] AS [Date],
            [{col_map['Open']}] AS [Open_],
            [{col_map['High']}] AS [High_],
            [{col_map['Low']}] AS [Low_],
            [{col_map['Close']}] AS [Close_],
            [{col_map['Volume']}] AS [Volume_]
        FROM [{table_name}]
        ORDER BY [{col_map['Date']}]
        """
        cursor.execute(query)
        columns = ['Date', 'Open_', 'High_', 'Low_', 'Close_', 'Volume_']
        data = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
        conn.close()

        # 轉回標準欄位名稱
        if not data.empty:
            data.rename(columns={
                'Open_': 'Open',
                'High_': 'High',
                'Low_': 'Low',
                'Close_': 'Close',
                'Volume_': 'Volume'
            }, inplace=True)

        # 若資料為空，回傳空 DataFrame
        if len(data) == 0:
            print(f"資料表 {table_name} 無資料。")
        return data
    except Exception as e:
        print(f"取得股票資料時發生錯誤: {str(e)}")
        return pd.DataFrame()

# === 計算股價區間 ===
def calculate_intervals(data, m):
    # 確保價格欄位為 float 型別
    data['High'] = pd.to_numeric(data['High'], errors='coerce')
    data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
    max_price = data['High'].max()
    min_price = data['Low'].min()
    interval_length = (max_price - min_price) / m
    intervals = [(min_price + i * interval_length, min_price + (i + 1) * interval_length) for i in range(m)]
    return intervals, interval_length

# === 計算利潤序對 ===
def calculate_profit(data, h_time, transaction_cost):
    # 確保收盤價為 float 型別
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    buy_sell_pairs = [(data['Close'].iloc[i], data['Close'].iloc[i + h_time])
                      for i in range(len(data) - h_time)]
    profit_pairs = [(buy, sell - buy - (buy * transaction_cost)) for buy, sell in buy_sell_pairs]
    return buy_sell_pairs, profit_pairs

# === 分析各區間獲利資訊 ===
def analyze_profit_by_interval(intervals, profit_pairs, target_profit, alpha):
    """
    分析每個價格區間的獲利情況
    
    參數:
    - intervals: 價格區間列表
    - profit_pairs: (買入價, 利潤) 配對列表
    - target_profit: 目標利潤閾值
    - alpha: 進場門檻 (目標利潤達成率門檻)
    """
    profit_interval_dict = {interval: [] for interval in intervals}
    
    # 將每筆交易的利潤分配到對應的價格區間
    for buy_price, profit in profit_pairs:
        for interval in intervals:
            if interval[0] <= buy_price < interval[1]:
                profit_interval_dict[interval].append(profit)
                break
    
    price_interval_trading_info = []
    for interval, profits in profit_interval_dict.items():
        if profits:  # 只處理有交易記錄的區間
            # 計算各種統計指標
            avg_profit = sum(profits) / len(profits)
            positive_profit_ratio = sum(1 for p in profits if p > 0) / len(profits)
            
            # 關鍵計算：目標利潤達成率
            target_profit_count = sum(1 for profit in profits if profit >= target_profit)
            t_profit_prob = target_profit_count / len(profits)
            
            # 決策邏輯：基於進場門檻 alpha 決定買賣訊號
            if t_profit_prob >= alpha:
                signal = "買入訊號"
            else:
                signal = "賣出訊號"
            
            price_interval_trading_info.append((
                interval, 
                avg_profit, 
                t_profit_prob, 
                signal, 
                len(profits),
                target_profit_count  # 新增：達成目標利潤的次數
            ))
        else:
            # 沒有交易記錄的區間標記為無資料
            price_interval_trading_info.append((
                interval, 
                0.0, 
                0.0, 
                "無交易資料", 
                0,
                0
            ))
    
    return price_interval_trading_info

# === 產生交易規則 ===
def generate_trading_rules(price_interval_trading_info, target_profit, hold_days, alpha):
    """
    根據區間分析結果產生具體的交易規則
    
    參數:
    - price_interval_trading_info: 區間分析結果
    - target_profit: 目標利潤金額
    - hold_days: 持有天數
    - alpha: 進場門檻
    """
    rules = []
    buy_intervals = []
    sell_intervals = []
    
    for i, info in enumerate(price_interval_trading_info):
        if len(info) == 6:  # 有交易資料
            interval, avg_profit, t_profit_prob, signal, sample_count, target_count = info
        else:  # 向後兼容
            interval, avg_profit, t_profit_prob, signal, sample_count = info
            target_count = int(t_profit_prob * sample_count) if sample_count > 0 else 0
        
        interval_desc = f"{interval[0]:.2f}~{interval[1]:.2f}"
        
        if signal == "買入訊號":
            buy_intervals.append(i + 1)
            buy_rule = f"💰 區間 {i+1} ({interval_desc}): 買入訊號"
            detail_rule = f"   • 持有期間: {hold_days} 天"
            profit_rule = f"   • 目標利潤: {target_profit:.2f} 元 (達成率 {t_profit_prob:.1%} ≥ {alpha:.1%})"
            stop_rule = f"   • 停損設定: 股價跌破 {interval[0] * 0.95:.2f} 元"
            stats_rule = f"   • 歷史統計: {target_count}/{sample_count} 次達標, 平均利潤 {avg_profit:.2f}"
            
            rules.append([buy_rule, detail_rule, profit_rule, stop_rule, stats_rule])
            
        elif signal == "賣出訊號":
            sell_intervals.append(i + 1)
            sell_rule = f"🚫 區間 {i+1} ({interval_desc}): 賣出訊號"
            detail_rule = f"   • 目標利潤達成率 {t_profit_prob:.1%} < {alpha:.1%} (不符進場條件)"
            stats_rule = f"   • 歷史統計: {target_count}/{sample_count} 次達標, 平均利潤 {avg_profit:.2f}"
            
            rules.append([sell_rule, detail_rule, stats_rule])
            
        else:  # 無交易資料
            no_data_rule = f"❓ 區間 {i+1} ({interval_desc}): 無歷史交易資料"
            rules.append([no_data_rule])
    
    # 添加策略總結
    summary_rules = [
        f"\n📋 策略總結:",
        f"   • 建議買入區間: {buy_intervals if buy_intervals else '無'}",
        f"   • 建議賣出區間: {sell_intervals if sell_intervals else '無'}",
        f"   • 持有期間: {hold_days} 天",
        f"   • 目標利潤: {target_profit:.2f} 元",
        f"   • 進場門檻: {alpha:.1%}"
    ]
    
    rules.append(summary_rules)
    
    return rules

# === 儲存資料到 SQL Server ===
def save_stock_data_to_db(data, table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 建立資料表（如果不存在）
    cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='StockPrices' AND xtype='U')
    CREATE TABLE StockPrices (
        TradeDate datetime,
        Ticker varchar(50),  -- 將 varchar(10) 改為 varchar(50)
        OpenPrice float,
        HighPrice float,
        LowPrice float,
        ClosePrice float,
        Volume bigint,
        PRIMARY KEY (TradeDate, Ticker)
    )
    """)
    
    # 插入資料
    for _, row in data.iterrows():
        cursor.execute("""
        IF NOT EXISTS (
            SELECT * FROM StockPrices 
            WHERE TradeDate = ? AND Ticker = ?
        )
        INSERT INTO StockPrices (
            TradeDate, Ticker, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row['Date'], str(table_name)[:50],  # 保證不超過50字元
            row['Date'], str(table_name)[:50],
            row['Open'], row['High'],
            row['Low'], row['Close'],
            row['Volume']
        ))
    
    conn.commit()
    conn.close()

def get_parameters_gradio():
    tables = get_available_tickers()
    return tables

def ppts_strategy(table_name, m, h_time, target_profit_ratio, alpha):
    """
    核心策略函數：根據參數動態生成交易策略
    
    參數說明:
    - table_name: 資料表名稱
    - m: 區間個數
    - h_time: 持有期間（天）⭐ 影響買賣對計算
    - target_profit_ratio: 目標利潤比例 ⭐ 影響目標利潤閾值
    - alpha: 進場門檻（目標利潤達成率閾值）⭐ 影響買賣訊號
    """
    # 參數轉換和驗證
    M = max(3, min(10, int(m)))
    HOLD_DAYS = max(1, min(60, int(h_time)))  # ⭐ 持有期間參數
    ALPHA = max(0.0, min(1.0, float(alpha)))  # ⭐ 進場門檻參數
    TARGET_PROFIT_RATIO = max(0.01, min(5.0, float(target_profit_ratio)))  # ⭐ 目標利潤比例參數
    TRANSACTION_COST = 0.001

    print(f"🔧 使用參數: h_time={HOLD_DAYS}, target_profit_ratio={TARGET_PROFIT_RATIO}, alpha={ALPHA}")

    # 獲取股票資料
    data = get_stock_data(table_name)
    if data.empty:
        return "⚠️ 無法取得有效資料，請檢查資料表內容或選擇其他表。", None

    # 檢查資料是否足夠進行分析
    if len(data) <= HOLD_DAYS:
        return f"⚠️ 資料不足：僅有 {len(data)} 筆資料，但需要至少 {HOLD_DAYS + 1} 筆資料進行 {HOLD_DAYS} 天持有期分析。", None

    # 計算價格區間
    intervals, interval_length = calculate_intervals(data, M)
    
    # ⭐ 關鍵修正1: 使用動態持有期間計算買賣配對
    print(f"📊 計算 {HOLD_DAYS} 天持有期間的買賣配對...")
    buy_sell_pairs, profit_pairs = calculate_profit(data, HOLD_DAYS, TRANSACTION_COST)
    
    # ⭐ 關鍵修正2: 使用動態目標利潤比例計算目標利潤
    target_profit = TARGET_PROFIT_RATIO * interval_length
    print(f"💰 目標利潤計算: {TARGET_PROFIT_RATIO} × {interval_length:.2f} = {target_profit:.2f}")
    
    # ⭐ 關鍵修正3: 傳遞動態alpha參數進行區間分析
    print(f"🎯 使用進場門檻 α={ALPHA} 進行區間分析...")
    trading_info = analyze_profit_by_interval(intervals, profit_pairs, target_profit, ALPHA)
    
    # 生成交易規則（傳入所有動態參數）
    rules = generate_trading_rules(trading_info, target_profit, HOLD_DAYS, ALPHA)

    # 計算統計摘要
    buy_signals = sum(1 for info in trading_info if len(info) > 3 and info[3] == "買入訊號")
    sell_signals = sum(1 for info in trading_info if len(info) > 3 and info[3] == "賣出訊號")
    total_trades = len(profit_pairs)
    
    # 生成詳細的參數報告 - 強調參數變化的影響
    rules_text = f"""
🔧 **動態參數設定** (參數變化會直接影響下方結果)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 資料表: {table_name}
📈 總資料筆數: {len(data)} 筆

⏰ **持有期間**: {HOLD_DAYS} 天 
   └─ 影響: 每筆交易持有 {HOLD_DAYS} 天後賣出，共產生 {total_trades} 個交易對

💰 **目標利潤設定**: 
   └─ 比例: {TARGET_PROFIT_RATIO:.2f}
   └─ 區間長度: {interval_length:.2f} 元
   └─ 實際目標利潤: {target_profit:.2f} 元
   └─ 影響: 利潤超過 {target_profit:.2f} 元才算達成目標

🎯 **進場門檻**: α = {ALPHA:.1%}
   └─ 影響: 目標利潤達成率需 ≥ {ALPHA:.1%} 才產生買入訊號
   └─ 結果: {buy_signals} 個買入區間, {sell_signals} 個賣出區間

📊 區間個數: {M} 個
🔄 交易成本: {TRANSACTION_COST:.1%}

🎯 **交易規則詳情** (基於上述參數動態計算)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # 添加規則詳情
    for i, rule_group in enumerate(rules):
        if isinstance(rule_group, list):
            for rule in rule_group:
                rules_text += f"{rule}\n"
        else:
            rules_text += f"{rule_group}\n"
        rules_text += "\n"

    # 添加參數敏感性說明
    rules_text += f"""
📈 **參數影響分析**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 調整持有期間 {HOLD_DAYS} → {HOLD_DAYS+5} 天: 會重新計算 {total_trades} 個交易對
• 調整目標利潤 {TARGET_PROFIT_RATIO:.1f} → {TARGET_PROFIT_RATIO+0.1:.1f}: 目標從 {target_profit:.2f} → {(TARGET_PROFIT_RATIO+0.1)*interval_length:.2f} 元
• 調整進場門檻 {ALPHA:.1%} → {ALPHA+0.05:.1%}: 可能改變買入區間數量
"""

    # 創建視覺化圖表
    try:
        # 確保使用中文字體
        plt.style.use('default')
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 確保日期格式正確
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # 圖1: 股價走勢與區間
        ax1.plot(data['Date'], data['Close'], label='收盤價', linewidth=2, color='navy')
        
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (interval, avg_profit, t_profit_prob, signal, sample_count, *_) in enumerate(trading_info):
            color = colors[i % len(colors)]
            alpha_val = 0.4 if signal == "買入訊號" else 0.2
            ax1.axhspan(interval[0], interval[1], color=color, alpha=alpha_val, 
                       label=f'區間{i+1}')
        
        ax1.set_title(f'{table_name} 股價走勢與交易區間\n(持有期間: {HOLD_DAYS}天, 目標利潤: {target_profit:.2f}元)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('股價 (元)', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 圖2: 目標利潤達成率比較 - 顯示參數影響
        interval_nums = range(1, len(trading_info) + 1)
        profit_probs = [info[2] for info in trading_info]
        signals = [info[3] for info in trading_info]
        
        colors_bar = ['green' if s == "買入訊號" else 'red' if s == "賣出訊號" else 'gray' 
                      for s in signals]
        
        bars = ax2.bar(interval_nums, profit_probs, color=colors_bar, alpha=0.7)
        ax2.axhline(y=ALPHA, color='orange', linestyle='--', linewidth=3, 
                    label=f'進場門檻 α={ALPHA:.1%}')
        
        ax2.set_xlabel('區間編號', fontsize=12)
        ax2.set_ylabel('目標利潤達成率', fontsize=12)
        ax2.set_title(f'目標利潤達成率 vs 進場門檻\n(目標利潤: {target_profit:.2f}元 = {TARGET_PROFIT_RATIO:.1f} × {interval_length:.2f})', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 在柱狀圖上標示數值和訊號
        for i, (bar, prob, signal) in enumerate(zip(bars, profit_probs, signals)):
            height = bar.get_height()
            signal_text = "買" if signal == "買入訊號" else "賣" if signal == "賣出訊號" else "無"
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    f'{prob:.1%}\n({signal_text})', ha='center', va='bottom', fontsize=9)
        
        # 圖3: 各區間平均利潤 vs 目標利潤
        avg_profits = [info[1] for info in trading_info]
        bars3 = ax3.bar(interval_nums, avg_profits, color=colors_bar, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, label='損益平衡線')
        ax3.axhline(y=target_profit, color='orange', linestyle='--', linewidth=3, 
                    label=f'目標利潤 {target_profit:.2f}元')
        
        ax3.set_xlabel('區間編號', fontsize=12)
        ax3.set_ylabel('平均利潤 (元)', fontsize=12)
        ax3.set_title(f'各區間平均利潤分析\n(持有期間: {HOLD_DAYS}天)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 標示數值
        for bar, profit in zip(bars3, avg_profits):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    height + (0.1 if height >= 0 else -0.2), 
                    f'{profit:.2f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # 圖4: 交易樣本數統計
        sample_counts = [info[4] for info in trading_info]
        bars4 = ax4.bar(interval_nums, sample_counts, color='lightblue', alpha=0.7)
        
        ax4.set_xlabel('區間編號', fontsize=12)
        ax4.set_ylabel('歷史交易次數', fontsize=12)
        ax4.set_title(f'各區間歷史交易樣本數\n(總計 {sum(sample_counts)} 次, 持有期間 {HOLD_DAYS} 天)', 
                     fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 在柱狀圖上標示數值
        for bar, count in zip(bars4, sample_counts):
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        return rules_text, fig
        
    except Exception as e:
        error_msg = f"""
{rules_text}

⚠️ 圖表生成錯誤: {str(e)}

可能的解決方案:
1. 檢查系統是否安裝中文字體
2. 嘗試重新啟動程序
3. 檢查資料是否完整
"""
        return error_msg, None

# Gradio 介面
tables = get_parameters_gradio()
with gr.Blocks(title="股價區間交易策略產生器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📈 股價區間交易策略產生器
    
    **功能說明**: 根據歷史股價資料，動態調整參數來產生不同的區間交易策略
    
    **重要提醒**: 調整下方任一參數都會重新計算交易規則，請觀察規則變化！
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 基本設定")
            table_dropdown = gr.Dropdown(
                choices=tables, 
                label="🗂️ 選擇資料表", 
                value=tables[0] if tables else None,
                info="選擇要分析的股票資料表"
            )
            
            m_slider = gr.Slider(
                3, 10, value=5, step=1, 
                label="📊 區間個數 (m)",
                info="將股價範圍分成幾個區間進行分析"
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 策略參數")
            htime_slider = gr.Slider(
                5, 30, value=10, step=1, 
                label="⏰ 持有期間 (天)",
                info="買入後持有幾天再賣出"
            )
            
            target_profit_slider = gr.Slider(
                0.1, 2.0, value=0.5, step=0.1, 
                label="🎯 目標利潤比例",
                info="目標利潤 = 比例 × 區間長度"
            )
            
            alpha_slider = gr.Slider(
                0.1, 0.9, value=0.6, step=0.05, 
                label="🚪 進場門檻 (α)",
                info="目標利潤達成率超過此值才產生買入訊號"
            )
    
    with gr.Row():
        btn = gr.Button(
            "🚀 重新計算交易策略", 
            variant="primary", 
            size="lg"
        )
        
    gr.Markdown("### 📊 分析結果")
    
    with gr.Row():
        with gr.Column(scale=2):
            rules_output = gr.Textbox(
                label="📋 詳細交易規則與參數分析", 
                lines=25,
                max_lines=30,
                show_copy_button=True
            )
        with gr.Column(scale=3):
            plot_output = gr.Plot(label="📈 策略分析圖表")

    # 參數變化時自動更新
    def update_strategy(*args):
        return ppts_strategy(*args)
    
    # 綁定所有輸入控件
    inputs = [table_dropdown, m_slider, htime_slider, target_profit_slider, alpha_slider]
    outputs = [rules_output, plot_output]
    
    # 按鈕點擊事件
    btn.click(fn=update_strategy, inputs=inputs, outputs=outputs)
    
    # 停用自動更新 (這裡是導致錯誤的主要原因)
    # 以下行為可能導致參數不足的問題，因為change事件可能不總是返回所有需要的參數
    # for input_component in inputs:
    #     if hasattr(input_component, 'change'):
    #         input_component.change(fn=update_strategy, inputs=inputs, outputs=outputs)
    
    # 改為只在按鈕點擊時更新
    
    # 頁面說明
    gr.Markdown("""
    ---
    ### 💡 使用說明
    
    1. **調整持有期間**: 改變 `⏰ 持有期間` 會影響買賣對的計算，進而改變每個區間的獲利統計
    2. **調整目標利潤**: 改變 `🎯 目標利潤比例` 會改變目標利潤閾值，影響達成率計算
    3. **調整進場門檻**: 改變 `🚪 進場門檻 α` 會直接影響哪些區間被標記為買入或賣出訊號
    4. **觀察變化**: 每次調整參數後，交易規則、統計數據和圖表都會相應更新
    
    ### 📈 圖表說明
    - **左上**: 股價走勢與區間劃分
    - **右上**: 各區間目標利潤達成率 vs 進場門檻  
    - **左下**: 各區間歷史平均利潤
    - **右下**: 各區間歷史交易樣本數
    
    """)

if __name__ == "__main__":
    # 確認環境信息
    print("系統平台:", platform.system())
    print("Python版本:", platform.python_version())
    
    # 字體檢查
    font_list = [f.name for f in fm.fontManager.ttflist if '微軟' in f.name or 'Microsoft' in f.name]
    print("可用中文字體:", font_list if font_list else "未找到適合的中文字體")
    
    # 啟動介面，關閉自動分享與調試模式
    demo.launch(share=False, debug=False)