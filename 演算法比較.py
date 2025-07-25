# -*- coding: utf-8 -*-
"""
Trading Strategy TSMC Backtest v18 - 增強版 (含算力資源分析)
新增功能：
1. 算力資源消耗監控
2. 時間複雜度分析
3. 記憶體使用量追蹤
4. GPU使用率監控 (如果可用)
5. 演算法效能基準測試
"""

import os
import subprocess
import sys
import urllib.request
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
import warnings
import time
import psutil
import threading
from collections import defaultdict
import gc
warnings.filterwarnings('ignore')

# 新增：算力資源監控類別
class ComputationalResourceMonitor:
    """算力資源監控器"""
    
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        
        # 嘗試載入GPU監控
        try:
            import GPUtil
            self.gpu_available = True
            self.gputil = GPUtil
        except ImportError:
            self.gpu_available = False
            print("⚠️ GPU監控不可用，將只監控CPU和記憶體")
    
    def start_monitoring(self, interval=0.5):
        """開始監控資源使用"""
        self.monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.timestamps = []
        
        def monitor_loop():
            start_time = time.time()
            while self.monitoring:
                current_time = time.time() - start_time
                
                # CPU使用率
                cpu_percent = psutil.cpu_percent()
                
                # 記憶體使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU使用率 (如果可用)
                gpu_percent = 0
                if self.gpu_available:
                    try:
                        gpus = self.gputil.getGPUs()
                        if gpus:
                            gpu_percent = gpus[0].load * 100
                    except:
                        pass
                
                self.timestamps.append(current_time)
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                self.gpu_usage.append(gpu_percent)
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def get_statistics(self):
        """取得監控統計"""
        if not self.timestamps:
            return {}
        
        stats = {
            'duration': max(self.timestamps) if self.timestamps else 0,
            'cpu_avg': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_max': np.max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'memory_max': np.max(self.memory_usage) if self.memory_usage else 0,
            'samples': len(self.timestamps)
        }
        
        if self.gpu_available and self.gpu_usage:
            stats.update({
                'gpu_avg': np.mean(self.gpu_usage),
                'gpu_max': np.max(self.gpu_usage)
            })
        
        return stats

# 新增：算力複雜度分析器
class ComputationalComplexityAnalyzer:
    """算力複雜度分析器"""
    
    def __init__(self):
        self.algorithm_profiles = {
            'ARIMA': {
                'time_complexity': 'O(n²)',
                'space_complexity': 'O(n)',
                'cpu_intensive': 'Medium',
                'memory_intensive': 'Low',
                'parallelizable': 'Limited',
                'estimated_flops': lambda n: n**2 * 10
            },
            'GARCH': {
                'time_complexity': 'O(n²)',
                'space_complexity': 'O(n)',
                'cpu_intensive': 'Medium',
                'memory_intensive': 'Low',
                'parallelizable': 'Limited',
                'estimated_flops': lambda n: n**2 * 15
            },
            'SVM': {
                'time_complexity': 'O(n³)',
                'space_complexity': 'O(n²)',
                'cpu_intensive': 'High',
                'memory_intensive': 'Medium',
                'parallelizable': 'Partial',
                'estimated_flops': lambda n: n**3 * 5
            },
            '隨機森林': {
                'time_complexity': 'O(n×log(n)×k)',
                'space_complexity': 'O(n×k)',
                'cpu_intensive': 'High',
                'memory_intensive': 'Medium',
                'parallelizable': 'High',
                'estimated_flops': lambda n: n * np.log(n) * 100 * 50
            },
            'XGBoost': {
                'time_complexity': 'O(n×log(n)×k)',
                'space_complexity': 'O(n×k)',
                'cpu_intensive': 'Very High',
                'memory_intensive': 'High',
                'parallelizable': 'High',
                'estimated_flops': lambda n: n * np.log(n) * 100 * 80
            },
            'LSTM': {
                'time_complexity': 'O(n×h²×layers)',
                'space_complexity': 'O(n×h×layers)',
                'cpu_intensive': 'Very High',
                'memory_intensive': 'Very High',
                'parallelizable': 'High (GPU)',
                'estimated_flops': lambda n: n * 256**2 * 3 * 4  # 假設256隱藏單元，3層
            },
            'CNN': {
                'time_complexity': 'O(n×k²×filters)',
                'space_complexity': 'O(n×filters)',
                'cpu_intensive': 'Very High',
                'memory_intensive': 'High',
                'parallelizable': 'Very High (GPU)',
                'estimated_flops': lambda n: n * 9 * 64 * 10  # 假設3x3卷積，64filters，10層
            },
            'Transformer': {
                'time_complexity': 'O(n²×d)',
                'space_complexity': 'O(n²+n×d)',
                'cpu_intensive': 'Extreme',
                'memory_intensive': 'Extreme',
                'parallelizable': 'Very High (GPU)',
                'estimated_flops': lambda n: n**2 * 512 * 12 * 2  # 假設512維度，12層，2倍overhead
            },
            '基因演算法': {
                'time_complexity': 'O(pop×gen×eval)',
                'space_complexity': 'O(pop×genes)',
                'cpu_intensive': 'High',
                'memory_intensive': 'Medium',
                'parallelizable': 'High',
                'estimated_flops': lambda n: 50 * 100 * n * 10  # 50個體，100世代
            }
        }
    
    def estimate_computational_cost(self, algorithm, data_size):
        """估算演算法的計算成本"""
        if algorithm not in self.algorithm_profiles:
            return None
        
        profile = self.algorithm_profiles[algorithm]
        estimated_flops = profile['estimated_flops'](data_size)
        
        # 估算執行時間 (假設1GFLOPS處理能力)
        estimated_time = estimated_flops / (1e9)  # 秒
        
        # 估算記憶體需求 (簡化計算)
        if 'O(n²)' in profile['space_complexity']:
            estimated_memory = data_size**2 * 8 / (1024**2)  # MB
        elif 'O(n)' in profile['space_complexity']:
            estimated_memory = data_size * 8 / (1024**2)  # MB
        else:
            estimated_memory = data_size * 100 / (1024**2)  # MB (預設)
        
        return {
            'estimated_flops': estimated_flops,
            'estimated_time_seconds': estimated_time,
            'estimated_memory_mb': estimated_memory,
            'profile': profile
        }
    
    def create_complexity_comparison(self, data_sizes=[100, 500, 1000, 2000]):
        """建立複雜度比較表"""
        results = []
        
        for algo_name in self.algorithm_profiles.keys():
            for size in data_sizes:
                cost = self.estimate_computational_cost(algo_name, size)
                if cost:
                    results.append({
                        '演算法': algo_name,
                        '資料大小': size,
                        '估算FLOPS': f"{cost['estimated_flops']:.2e}",
                        '估算時間(秒)': f"{cost['estimated_time_seconds']:.2f}",
                        '估算記憶體(MB)': f"{cost['estimated_memory_mb']:.1f}",
                        'CPU密集度': cost['profile']['cpu_intensive'],
                        '可平行化': cost['profile']['parallelizable']
                    })
        
        return pd.DataFrame(results)

# 修改主要函數，加入資源監控
def main():
    """主要執行函數 - 增強版"""
    print("🚀 開始執行 TSMC 交易策略回測系統 (增強版)...")
    
    # 初始化資源監控器
    resource_monitor = ComputationalResourceMonitor()
    complexity_analyzer = ComputationalComplexityAnalyzer()
    
    # 開始系統級監控
    resource_monitor.start_monitoring()
    main_start_time = time.time()
    
    try:
        # 安裝必要套件
        install_packages()
        
        # 設定中文字型
        zh_font = setup_chinese_font()
        
        # 載入其他必要模組 (計時)
        module_start = time.time()
        load_required_modules()
        module_time = time.time() - module_start
        print(f'✅ 模組載入時間: {module_time:.2f} 秒')
        
        # 下載資料 (計時)
        data_start = time.time()
        data, train, test, price_data = download_tsmc_data()
        data_time = time.time() - data_start
        print(f'✅ 資料下載時間: {data_time:.2f} 秒')
        
        if data is None:
            return
        
        # 建立演算法比較表
        algo_df = create_algorithm_dataframe()
        print("\n📊 演算法比較表:")
        from tabulate import tabulate
        print(tabulate(algo_df, headers='keys', tablefmt='grid'))
        
        # 建立複雜度分析表
        print("\n🧮 算力複雜度分析:")
        complexity_df = complexity_analyzer.create_complexity_comparison()
        print(tabulate(complexity_df, headers='keys', tablefmt='grid'))
        
        # 建立演算法比較圖表
        create_algorithm_comparison_chart(algo_df, zh_font)
        
        # 建立複雜度分析圖表
        create_complexity_analysis_chart(complexity_analyzer, zh_font)
        
        # 顯示股價資料基本統計
        display_stock_statistics(price_data)
        
        # 繪製股價走勢圖
        create_price_chart(price_data, zh_font)
        
        # 執行基因演算法預測 (帶資源監控)
        print("\n🧬 開始執行基因演算法股價預測...")
        ga_start = time.time()
        
        # 為基因演算法創建專用監控器
        ga_monitor = ComputationalResourceMonitor()
        ga_monitor.start_monitoring()
        
        ga_result = genetic_algorithm_prediction(train, test, zh_font)
        
        ga_monitor.stop_monitoring()
        ga_time = time.time() - ga_start
        ga_stats = ga_monitor.get_statistics()
        
        print(f"✅ 基因演算法完成:")
        print(f"  - 執行時間: {ga_time:.2f} 秒")
        print(f"  - 平均CPU使用率: {ga_stats.get('cpu_avg', 0):.1f}%")
        print(f"  - 最大CPU使用率: {ga_stats.get('cpu_max', 0):.1f}%")
        print(f"  - 平均記憶體使用率: {ga_stats.get('memory_avg', 0):.1f}%")
        print(f"  - RMSE: {ga_result['rmse']:.4f}")
        
        if ga_stats.get('gpu_avg', 0) > 0:
            print(f"  - 平均GPU使用率: {ga_stats.get('gpu_avg', 0):.1f}%")
        
        # 多股票分析 (可選)
        stocks = {
            'TSMC': '2330.TW',
            '台積電ADR': 'TSM',
            '聯發科': '2454.TW',
            '鴻海': '2317.TW'
        }
        
        print("\n📈 下載多股票資料進行比較分析...")
        multi_start = time.time()
        stock_data = download_multiple_stocks(stocks)
        multi_time = time.time() - multi_start
        print(f'✅ 多股票資料下載時間: {multi_time:.2f} 秒')
        
        if stock_data:
            create_multi_stock_analysis(stock_data, zh_font)
        
        # 總執行時間統計
        total_time = time.time() - main_start_time
        resource_monitor.stop_monitoring()
        overall_stats = resource_monitor.get_statistics()
        
        # 建立資源使用統計圖表
        create_resource_usage_chart(resource_monitor, zh_font)
        
        # 顯示總體統計
        print("\n📊 總體執行統計:")
        print(f"總執行時間: {total_time:.2f} 秒")
        print(f"平均CPU使用率: {overall_stats.get('cpu_avg', 0):.1f}%")
        print(f"最大CPU使用率: {overall_stats.get('cpu_max', 0):.1f}%")
        print(f"平均記憶體使用率: {overall_stats.get('memory_avg', 0):.1f}%")
        print(f"最大記憶體使用率: {overall_stats.get('memory_max', 0):.1f}%")
        
        if overall_stats.get('gpu_avg', 0) > 0:
            print(f"平均GPU使用率: {overall_stats.get('gpu_avg', 0):.1f}%")
            print(f"最大GPU使用率: {overall_stats.get('gpu_max', 0):.1f}%")
        
        # 算力資源建議
        provide_computational_recommendations(overall_stats, total_time)
        
    except Exception as e:
        print(f"❌ 執行過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        resource_monitor.stop_monitoring()
    
    print("\n🎉 TSMC 交易策略回測系統 (增強版) 執行完成！")

def create_complexity_analysis_chart(complexity_analyzer, zh_font):
    """建立複雜度分析圖表"""
    data_sizes = [100, 500, 1000, 2000, 5000]
    algorithms = ['ARIMA', 'SVM', '隨機森林', 'XGBoost', 'LSTM', 'CNN', 'Transformer', '基因演算法']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子圖1: 執行時間比較
    for algo in algorithms:
        times = []
        for size in data_sizes:
            cost = complexity_analyzer.estimate_computational_cost(algo, size)
            times.append(cost['estimated_time_seconds'] if cost else 0)
        ax1.plot(data_sizes, times, marker='o', linewidth=2, label=algo)
    
    ax1.set_title('估算執行時間 vs 資料大小', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax1.set_xlabel('資料大小', fontproperties=zh_font, fontsize=12)
    ax1.set_ylabel('時間 (秒)', fontproperties=zh_font, fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(prop=zh_font, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子圖2: 記憶體需求比較
    for algo in algorithms:
        memories = []
        for size in data_sizes:
            cost = complexity_analyzer.estimate_computational_cost(algo, size)
            memories.append(cost['estimated_memory_mb'] if cost else 0)
        ax2.plot(data_sizes, memories, marker='s', linewidth=2, label=algo)
    
    ax2.set_title('估算記憶體需求 vs 資料大小', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax2.set_xlabel('資料大小', fontproperties=zh_font, fontsize=12)
    ax2.set_ylabel('記憶體 (MB)', fontproperties=zh_font, fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(prop=zh_font, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子圖3: FLOPS 比較
    for algo in algorithms:
        flops = []
        for size in data_sizes:
            cost = complexity_analyzer.estimate_computational_cost(algo, size)
            flops.append(cost['estimated_flops'] if cost else 0)
        ax3.plot(data_sizes, flops, marker='^', linewidth=2, label=algo)
    
    ax3.set_title('估算浮點運算量 (FLOPS) vs 資料大小', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax3.set_xlabel('資料大小', fontproperties=zh_font, fontsize=12)
    ax3.set_ylabel('FLOPS', fontproperties=zh_font, fontsize=12)
    ax3.set_yscale('log')
    ax3.legend(prop=zh_font, fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子圖4: CPU密集度矩陣
    cpu_intensity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4, 'Extreme': 5}
    parallelizable_map = {'Limited': 1, 'Partial': 2, 'High': 3, 'Very High (GPU)': 4, 'High (GPU)': 4}
    
    algo_profiles = complexity_analyzer.algorithm_profiles
    x_vals = []
    y_vals = []
    labels = []
    
    for algo, profile in algo_profiles.items():
        cpu_score = cpu_intensity_map.get(profile['cpu_intensive'], 2)
        parallel_score = parallelizable_map.get(profile['parallelizable'], 2)
        x_vals.append(cpu_score)
        y_vals.append(parallel_score)
        labels.append(algo)
    
    scatter = ax4.scatter(x_vals, y_vals, s=100, alpha=0.7, c=range(len(labels)), cmap='tab10')
    
    for i, label in enumerate(labels):
        ax4.annotate(label, (x_vals[i], y_vals[i]), xytext=(5, 5), 
                    textcoords='offset points', fontproperties=zh_font, fontsize=10)
    
    ax4.set_title('CPU密集度 vs 可平行化程度', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax4.set_xlabel('CPU密集度', fontproperties=zh_font, fontsize=12)
    ax4.set_ylabel('可平行化程度', fontproperties=zh_font, fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    complexity_file = 'computational_complexity_analysis.png'
    plt.savefig(complexity_file, dpi=300, bbox_inches='tight')
    print(f"✅ 複雜度分析圖已儲存為: {complexity_file}")
    plt.show()

def create_resource_usage_chart(monitor, zh_font):
    """建立資源使用圖表"""
    if not monitor.timestamps:
        print("⚠️ 無資源監控資料可顯示")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # CPU使用率時間序列
    axes[0,0].plot(monitor.timestamps, monitor.cpu_usage, color='blue', linewidth=2)
    axes[0,0].set_title('CPU使用率時間序列', fontproperties=zh_font, fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('時間 (秒)', fontproperties=zh_font, fontsize=12)
    axes[0,0].set_ylabel('CPU使用率 (%)', fontproperties=zh_font, fontsize=12)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 100)
    
    # 記憶體使用率時間序列
    axes[0,1].plot(monitor.timestamps, monitor.memory_usage, color='green', linewidth=2)
    axes[0,1].set_title('記憶體使用率時間序列', fontproperties=zh_font, fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('時間 (秒)', fontproperties=zh_font, fontsize=12)
    axes[0,1].set_ylabel('記憶體使用率 (%)', fontproperties=zh_font, fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 100)
    
    # 資源使用分布
    axes[1,0].hist(monitor.cpu_usage, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1,0].set_title('CPU使用率分布', fontproperties=zh_font, fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('CPU使用率 (%)', fontproperties=zh_font, fontsize=12)
    axes[1,0].set_ylabel('頻率', fontproperties=zh_font, fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # GPU使用率 (如果有的話)
    if monitor.gpu_available and any(monitor.gpu_usage):
        axes[1,1].plot(monitor.timestamps, monitor.gpu_usage, color='red', linewidth=2)
        axes[1,1].set_title('GPU使用率時間序列', fontproperties=zh_font, fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('時間 (秒)', fontproperties=zh_font, fontsize=12)
        axes[1,1].set_ylabel('GPU使用率 (%)', fontproperties=zh_font, fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_ylim(0, 100)
    else:
        # 記憶體使用分布
        axes[1,1].hist(monitor.memory_usage, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].set_title('記憶體使用率分布', fontproperties=zh_font, fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('記憶體使用率 (%)', fontproperties=zh_font, fontsize=12)
        axes[1,1].set_ylabel('頻率', fontproperties=zh_font, fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    resource_file = 'resource_usage_analysis.png'
    plt.savefig(resource_file, dpi=300, bbox_inches='tight')
    print(f"✅ 資源使用分析圖已儲存為: {resource_file}")
    plt.show()

def provide_computational_recommendations(stats, total_time):
    """提供算力資源建議"""
    print("\n💡 算力資源優化建議:")
    
    cpu_avg = stats.get('cpu_avg', 0)
    memory_avg = stats.get('memory_avg', 0)
    gpu_avg = stats.get('gpu_avg', 0)
    
    recommendations = []
    
    # CPU建議
    if cpu_avg > 80:
        recommendations.append("🔥 CPU使用率偏高，建議：")
        recommendations.append("   • 考慮使用多核心並行處理")
        recommendations.append("   • 使用更高效能的CPU")
        recommendations.append("   • 優化演算法實作或使用近似演算法")
    elif cpu_avg < 30:
        recommendations.append("💚 CPU使用率良好，可考慮：")
        recommendations.append("   • 增加模型複雜度以提升預測精度")
        recommendations.append("   • 同時執行多個演算法進行比較")
    
    # 記憶體建議
    if memory_avg > 80:
        recommendations.append("🔥 記憶體使用率偏高，建議：")
        recommendations.append("   • 增加系統記憶體")
        recommendations.append("   • 使用資料批次處理")
        recommendations.append("   • 考慮使用記憶體映射檔案")
    
    # GPU建議
    if gpu_avg == 0:
        recommendations.append("🚀 GPU加速建議：")
        recommendations.append("   • 安裝CUDA相容的GPU")
        recommendations.append("   • 使用TensorFlow-GPU或PyTorch-GPU")
        recommendations.append("   • 深度學習模型可獲得10-100倍速度提升")
    elif gpu_avg < 50:
        recommendations.append("⚡ GPU使用率可提升：")
        recommendations.append("   • 增加batch size")
        recommendations.append("   • 使用更複雜的模型架構")
    
    # 執行時間建議
    if total_time > 300:  # 超過5分鐘
        recommendations.append("⏰ 執行時間優化建議：")
        recommendations.append("   • 使用預訓練模型")
        recommendations.append("   • 實施早停機制")
        recommendations.append("   • 考慮使用雲端運算資源")
    
    # 顯示建議
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ 目前的資源配置良好！")
    
    # 硬體升級建議
    print("\n🔧 硬體升級優先順序建議：")
    if memory_avg > 70:
        print("1. 💾 增加記憶體 (RAM) - 高優先級")
    if cpu_avg > 70:
        print("2. 🖥️  升級CPU - 中高優先級")
    if gpu_avg == 0:
        print("3. 🎮 添加GPU - 深度學習場景高優先級")
    if total_time > 600:
        print("4. 💿 升級到SSD - 資料IO優化")

# 以下是原有函數的簡化版本，保持核心功能
def install_packages():
    """安裝必要的套件"""
    packages = ['arch', 'yfinance', 'xgboost', 'tensorflow', 'tabulate', 'scikit-learn', 'psutil']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"正在安裝 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_chinese_font():
    """設定matplotlib中文字型"""
    try:
        # Windows 系統
        if os.name == 'nt':
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
        # Linux/Mac 系統
        else:
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False
        zh_font = fm.FontProperties()
        print('✅ 已設定系統中文字型')
    except Exception as e:
        print(f"❌ 字型設定失敗: {e}")
        zh_font = fm.FontProperties()
    
    return zh_font

def load_required_modules():
    """載入必要模組"""
    global RandomForestRegressor, XGBRegressor, SVR, ARIMA, arch_model
    global ExponentialSmoothing, Sequential, LSTM_Layer, Dense, Conv1D, Flatten, Adam, tf, tabulate
    
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.svm import SVR
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from arch import arch_model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        print("正在安裝統計模型套件...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels", "arch"])
        from statsmodels.tsa.arima.model import ARIMA
        from arch import arch_model
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM as LSTM_Layer, Conv1D, Flatten
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf
    except ImportError:
        print("正在安裝 TensorFlow...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM as LSTM_Layer, Conv1D, Flatten
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf
    
    from tabulate import tabulate

def download_tsmc_data():
    """下載TSMC資料"""
    print("📈 正在下載 TSMC 股價資料...")
    try:
        data = yf.download('2330.TW', start='2019-01-01', end='2024-12-31', progress=False)
        if data.empty:
            print("❌ 無法下載資料，請檢查網路連線")
            return None, None, None, None
        
        price_data = data['Close'].dropna()
        train = price_data['2019-01-01':'2023-12-31']
        test = price_data['2024-01-01':'2024-12-31']
        
        if len(train) == 0 or len(test) == 0:
            print("❌ 資料分割錯誤，請檢查日期範圍")
            return None, None, None, None
            
        print(f'✅ 資料載入完成 - 訓練資料筆數: {len(train)}, 測試資料筆數: {len(test)}')
        return data, train, test, price_data
        
    except Exception as e:
        print(f"❌ 資料下載失敗: {e}")
        return None, None, None, None

def create_algorithm_dataframe():
    """建立演算法比較表"""
    return pd.DataFrame({
        '演算法': ['ARIMA','指數平滑法','GARCH','SVM','隨機森林','XGBoost','RNN/LSTM','CNN','Transformer','基因演算法'],
        '預測精度': ['中','低-中','中(波動)','中-高','高','高','高','中-高','高','中-高'],
        '計算複雜度': ['低','極低','中','中','高','高','極高','高','極高','高'],
        '適用資料規模': ['小-中','小','小-中','中','中-大','中-大','大','大','大','中-大'],
        '解釋性': ['高','高','中-高','低','低','極低','極低','極低','極低','中'],
        '對非線性捕捉': ['低','低','中','中高','高','極高','極高','極高','極高','高'],
        '算力需求': ['低','極低','中','中-高','高','極高','極高','極高','極高','中-高'],
        '記憶體需求': ['低','極低','低','中','中-高','高','極高','高','極高','中'],
        '可平行化': ['限制','限制','限制','部分','高','高','極高(GPU)','極高(GPU)','極高(GPU)','高']
    })

def create_algorithm_comparison_chart(algo_df, zh_font):
    """建立演算法比較圖表"""
    complexity_map = {'極低':1,'低':2,'中':3,'高':4,'極高':5}
    accuracy_map = {'低-中':1.5,'中':2,'中-高':3,'高':4}
    compute_map = {'極低':1,'低':2,'中':3,'中-高':3.5,'高':4,'極高':5}
    
    algo_df['複雜度_score'] = algo_df['計算複雜度'].map(complexity_map)
    algo_df['精度_score'] = algo_df['預測精度'].map(accuracy_map)
    algo_df['算力_score'] = algo_df['算力需求'].map(compute_map)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原有的預測精度 vs 計算複雜度
    colors = plt.cm.Set3(np.linspace(0, 1, len(algo_df)))
    
    for i, (idx, row) in enumerate(algo_df.iterrows()):
        ax1.scatter(row['複雜度_score'], row['精度_score'], 
                   s=120, alpha=0.8, color=colors[i], edgecolors='black', linewidth=1)
        ax1.annotate(row['演算法'], (row['複雜度_score'], row['精度_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontproperties=zh_font, fontsize=10)
    
    ax1.set_title('預測精度 vs 計算複雜度', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax1.set_xlabel('計算複雜度', fontproperties=zh_font, fontsize=12)
    ax1.set_ylabel('預測精度', fontproperties=zh_font, fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 新增：算力需求 vs 預測精度
    for i, (idx, row) in enumerate(algo_df.iterrows()):
        ax2.scatter(row['算力_score'], row['精度_score'], 
                   s=120, alpha=0.8, color=colors[i], edgecolors='black', linewidth=1)
        ax2.annotate(row['演算法'], (row['算力_score'], row['精度_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontproperties=zh_font, fontsize=10)
    
    ax2.set_title('算力需求 vs 預測精度', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax2.set_xlabel('算力需求', fontproperties=zh_font, fontsize=12)
    ax2.set_ylabel('預測精度', fontproperties=zh_font, fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 記憶體需求比較
    memory_map = {'極低':1,'低':2,'中':3,'中-高':3.5,'高':4,'極高':5}
    memory_scores = [memory_map.get(x, 3) for x in algo_df['記憶體需求']]
    
    bars = ax3.bar(range(len(algo_df)), memory_scores, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('記憶體需求比較', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax3.set_xlabel('演算法', fontproperties=zh_font, fontsize=12)
    ax3.set_ylabel('記憶體需求等級', fontproperties=zh_font, fontsize=12)
    ax3.set_xticks(range(len(algo_df)))
    ax3.set_xticklabels(algo_df['演算法'], rotation=45, fontproperties=zh_font, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 可平行化程度比較
    parallel_map = {'限制':1,'部分':2,'高':3,'極高(GPU)':4}
    parallel_scores = [parallel_map.get(x, 2) for x in algo_df['可平行化']]
    
    bars = ax4.bar(range(len(algo_df)), parallel_scores, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_title('可平行化程度比較', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax4.set_xlabel('演算法', fontproperties=zh_font, fontsize=12)
    ax4.set_ylabel('可平行化等級', fontproperties=zh_font, fontsize=12)
    ax4.set_xticks(range(len(algo_df)))
    ax4.set_xticklabels(algo_df['演算法'], rotation=45, fontproperties=zh_font, fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = 'enhanced_algo_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 增強版演算法比較圖已儲存為: {output_file}")
    plt.show()

def display_stock_statistics(price_data):
    """顯示股價基本統計"""
    print(f"\n📈 TSMC 股價基本統計 (2019-2024):")
    print(f"最高價: NT$ {float(price_data.max()):.2f}")
    print(f"最低價: NT$ {float(price_data.min()):.2f}")
    print(f"平均價: NT$ {float(price_data.mean()):.2f}")
    print(f"標準差: NT$ {float(price_data.std()):.2f}")
    print(f"資料點數: {len(price_data)}")
    print(f"資料期間: {price_data.index[0].strftime('%Y-%m-%d')} 至 {price_data.index[-1].strftime('%Y-%m-%d')}")

def create_price_chart(price_data, zh_font):
    """繪製股價走勢圖"""
    plt.figure(figsize=(14,8))
    plt.plot(price_data.index, price_data.values, linewidth=2, color='#2E8B57')
    plt.title('TSMC (2330.TW) 股價走勢 2019-2024', fontproperties=zh_font, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('日期', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.ylabel('收盤價 (NT$)', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 標示訓練/測試期間
    plt.axvline(x=pd.to_datetime('2024-01-01'), color='red', linestyle='--', alpha=0.8, linewidth=2, label='測試期間開始')
    plt.legend(prop=zh_font, fontsize=12)
    
    price_chart_file = 'tsmc_price_chart.png'
    plt.savefig(price_chart_file, dpi=300, bbox_inches='tight')
    print(f"✅ 股價走勢圖已儲存為: {price_chart_file}")
    plt.show()

# 基因演算法相關類別和函數保持不變
class GeneticAlgorithmPredictor:
    """基因演算法股價預測器"""
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_chromosome = None
        self.best_fitness = float('inf')
        self.computation_stats = {
            'total_evaluations': 0,
            'total_mutations': 0,
            'total_crossovers': 0,
            'convergence_generation': None
        }
    
    def create_chromosome(self, length=10):
        """創建染色體 (權重向量)"""
        return np.random.uniform(-1, 1, length)
    
    def create_population(self, chromosome_length):
        """創建初始族群"""
        return [self.create_chromosome(chromosome_length) for _ in range(self.population_size)]
    
    def fitness_function(self, chromosome, X, y):
        """適應度函數 (使用RMSE)"""
        try:
            self.computation_stats['total_evaluations'] += 1
            predictions = np.dot(X, chromosome)
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            return rmse
        except:
            return float('inf')
    
    def selection(self, population, fitness_scores):
        """選擇操作 (錦標賽選擇)"""
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def crossover(self, parent1, parent2):
        """交配操作 (單點交配)"""
        self.computation_stats['total_crossovers'] += 1
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def mutation(self, chromosome):
        """突變操作"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                self.computation_stats['total_mutations'] += 1
                mutated[i] += np.random.normal(0, 0.1)
        return mutated
    
    def evolve(self, X_train, y_train):
        """演化過程"""
        chromosome_length = X_train.shape[1]
        population = self.create_population(chromosome_length)
        
        fitness_history = []
        best_fitness_threshold = float('inf')
        convergence_counter = 0
        
        for generation in range(self.generations):
            # 計算適應度
            fitness_scores = []
            for chromosome in population:
                fitness = self.fitness_function(chromosome, X_train, y_train)
                fitness_scores.append(fitness)
            
            # 記錄最佳適應度
            min_fitness = min(fitness_scores)
            fitness_history.append(min_fitness)
            
            # 檢查收斂
            if min_fitness < best_fitness_threshold * 0.995:  # 改善小於0.5%視為收斂
                best_fitness_threshold = min_fitness
                convergence_counter = 0
            else:
                convergence_counter += 1
            
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                best_idx = fitness_scores.index(min_fitness)
                self.best_chromosome = population[best_idx].copy()
                
                if self.computation_stats['convergence_generation'] is None and convergence_counter == 0:
                    self.computation_stats['convergence_generation'] = generation
            
            # 早停機制
            if convergence_counter > 20:
                print(f"演算法在第 {generation} 代收斂，提早停止")
                break
            
            # 選擇、交配、突變
            selected = self.selection(population, fitness_scores)
            new_population = []
            
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[(i+1) % len(selected)]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            if generation % 20 == 0:
                print(f"第 {generation} 代，最佳適應度: {min_fitness:.4f}")
        
        return fitness_history
    
    def predict(self, X_test):
        """使用最佳染色體進行預測"""
        if self.best_chromosome is None:
            raise ValueError("模型尚未訓練")
        return np.dot(X_test, self.best_chromosome)
    
    def get_computation_stats(self):
        """取得計算統計"""
        return self.computation_stats

def create_features(data, window_size=10):
    """創建特徵矩陣 - 增強版"""
    features = []
    targets = []
    
    for i in range(window_size, len(data)):
        # 使用過去window_size天的價格作為特徵
        feature = data.iloc[i-window_size:i].values
        target = data.iloc[i]
        
        # 添加技術指標特徵
        prices = data.iloc[i-window_size:i]
        sma = prices.mean()  # 簡單移動平均
        volatility = prices.std()  # 波動率
        
        # 價格變化率特徵
        if len(prices) > 1:
            price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            max_price = prices.max()
            min_price = prices.min()
            price_range = (max_price - min_price) / prices.mean()
        else:
            price_change = 0
            price_range = 0
        
        # 組合特徵
        combined_feature = np.concatenate([feature, [sma, volatility, price_change, price_range]])
        features.append(combined_feature)
        targets.append(target)
    
    return np.array(features), np.array(targets)

def genetic_algorithm_prediction(train_data, test_data, zh_font):
    """執行基因演算法預測 - 增強版"""
    
    # 準備訓練資料
    X_train, y_train = create_features(train_data, window_size=10)
    X_test, y_test = create_features(test_data, window_size=10)
    
    print(f"訓練特徵維度: {X_train.shape}")
    print(f"測試特徵維度: {X_test.shape}")
    
    # 檢查資料是否足夠
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ 資料不足，無法進行預測")
        return {'rmse': 0, 'mae': 0, 'predictions': [], 'actual': [], 'fitness_history': []}
    
    # 初始化基因演算法 - 調整參數以適應算力分析
    ga = GeneticAlgorithmPredictor(population_size=30, generations=100, mutation_rate=0.15)
    
    # 訓練模型
    print("🧬 開始基因演算法演化...")
    evolution_start = time.time()
    fitness_history = ga.evolve(X_train, y_train)
    evolution_time = time.time() - evolution_start
    
    # 取得計算統計
    comp_stats = ga.get_computation_stats()
    
    # 預測
    prediction_start = time.time()
    predictions = ga.predict(X_test)
    prediction_time = time.time() - prediction_start
    
    # 計算誤差
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    mae = np.mean(np.abs(y_test - predictions))
    
    # 顯示計算統計
    print(f"\n🧮 基因演算法計算統計:")
    print(f"演化時間: {evolution_time:.2f} 秒")
    print(f"預測時間: {prediction_time:.3f} 秒")
    print(f"總適應度評估次數: {comp_stats['total_evaluations']:,}")
    print(f"總交配次數: {comp_stats['total_crossovers']:,}")
    print(f"總突變次數: {comp_stats['total_mutations']:,}")
    if comp_stats['convergence_generation']:
        print(f"收斂於第 {comp_stats['convergence_generation']} 代")
    
    # 估算計算強度
    flops_per_evaluation = X_train.shape[1] * X_train.shape[0] * 2  # 矩陣乘法 + RMSE計算
    total_flops = comp_stats['total_evaluations'] * flops_per_evaluation
    flops_per_second = total_flops / evolution_time if evolution_time > 0 else 0
    
    print(f"估算浮點運算量: {total_flops:,.0f} FLOPS")
    print(f"平均運算速度: {flops_per_second/1e6:.1f} MFLOPS")
    
    # 繪製結果 - 增強版
    plt.figure(figsize=(18, 12))
    
    # 子圖1: 適應度演化過程
    plt.subplot(2, 3, 1)
    plt.plot(fitness_history, linewidth=2, color='red')
    plt.title('基因演算法適應度演化', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.xlabel('世代', fontproperties=zh_font, fontsize=12)
    plt.ylabel('RMSE', fontproperties=zh_font, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 子圖2: 預測結果比較
    plt.subplot(2, 3, 2)
    test_dates = test_data.index[10:]
    plt.plot(test_dates, y_test, label='實際價格', linewidth=2, color='blue')
    plt.plot(test_dates, predictions, label='GA預測', linewidth=2, color='red', linestyle='--')
    plt.title('基因演算法預測結果', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.xlabel('日期', fontproperties=zh_font, fontsize=12)
    plt.ylabel('股價 (NT$)', fontproperties=zh_font, fontsize=12)
    plt.legend(prop=zh_font, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子圖3: 預測誤差分布
    plt.subplot(2, 3, 3)
    errors = y_test - predictions
    plt.hist(errors, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.title('預測誤差分布', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.xlabel('誤差 (NT$)', fontproperties=zh_font, fontsize=12)
    plt.ylabel('頻率', fontproperties=zh_font, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 子圖4: 散點圖比較
    plt.subplot(2, 3, 4)
    plt.scatter(y_test, predictions, alpha=0.6, color='purple')
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title('預測 vs 實際散點圖', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.xlabel('實際價格 (NT$)', fontproperties=zh_font, fontsize=12)
    plt.ylabel('預測價格 (NT$)', fontproperties=zh_font, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 子圖5: 計算複雜度分析
    plt.subplot(2, 3, 5)
    generations = list(range(len(fitness_history)))
    cumulative_evaluations = [(i+1) * ga.population_size for i in generations]
    plt.plot(generations, cumulative_evaluations, linewidth=2, color='orange')
    plt.title('累積適應度評估次數', fontproperties=zh_font, fontsize=14, fontweight='bold')
    plt.xlabel('世代', fontproperties=zh_font, fontsize=12)
    plt.ylabel('累積評估次數', fontproperties=zh_font, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 子圖6: 算力效率分析
    plt.subplot(2, 3, 6)
    if len(fitness_history) > 1:
        efficiency = [(fitness_history[0] - f) / (i+1) for i, f in enumerate(fitness_history)]
        plt.plot(generations, efficiency, linewidth=2, color='brown')
        plt.title('算力效率 (誤差改善/世代)', fontproperties=zh_font, fontsize=14, fontweight='bold')
        plt.xlabel('世代', fontproperties=zh_font, fontsize=12)
        plt.ylabel('效率', fontproperties=zh_font, fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 儲存基因演算法結果圖
    ga_result_file = 'enhanced_genetic_algorithm_results.png'
    plt.savefig(ga_result_file, dpi=300, bbox_inches='tight')
    print(f"✅ 增強版基因演算法結果圖已儲存為: {ga_result_file}")
    plt.show()
    
    # 計算並顯示詳細誤差統計
    errors = y_test - predictions
    mape_values = []
    for actual, pred in zip(y_test, predictions):
        if actual != 0:
            mape_values.append(abs((actual - pred) / actual))
    mape = np.mean(mape_values) * 100 if mape_values else 0
    
    print(f"\n📊 基因演算法預測結果統計:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"平均絕對百分比誤差 (MAPE): {mape:.2f}%")
    print(f"最大誤差: {float(np.max(np.abs(errors))):.4f}")
    print(f"誤差標準差: {float(np.std(errors)):.4f}")
    print(f"R²決定係數: {1 - np.var(errors) / np.var(y_test):.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'actual': y_test,
        'fitness_history': fitness_history,
        'computation_stats': comp_stats,
        'evolution_time': evolution_time,
        'prediction_time': prediction_time,
        'total_flops': total_flops,
        'flops_per_second': flops_per_second
    }

def download_multiple_stocks(stocks):
    """下載多支股票資料 - 增強版"""
    stock_data = {}
    failed_stocks = []
    
    print("正在平行下載多支股票資料...")
    download_start = time.time()
    
    for stock_name, stock_code in stocks.items():
        try:
            print(f"正在下載 {stock_name} ({stock_code})...")
            stock_start = time.time()
            
            data = yf.download(stock_code, start='2019-01-01', end='2024-12-31', progress=False)
            stock_time = time.time() - stock_start
            
            if not data.empty and 'Close' in data.columns:
                price_data = data['Close'].dropna()
                if len(price_data) > 100:  # 確保有足夠的資料點
                    stock_data[stock_name] = price_data
                    print(f"✅ {stock_name}: {len(price_data)} 個資料點 ({stock_time:.1f}秒)")
                else:
                    failed_stocks.append(stock_name)
                    print(f"⚠️ {stock_name}: 資料點不足")
            else:
                failed_stocks.append(stock_name)
                print(f"❌ {stock_name}: 無法取得資料")
                
        except Exception as e:
            failed_stocks.append(stock_name)
            print(f"❌ {stock_name}: 下載失敗 - {e}")
    
    download_time = time.time() - download_start
    print(f"總下載時間: {download_time:.1f} 秒")
    
    if failed_stocks:
        print(f"\n⚠️ 以下股票下載失敗: {', '.join(failed_stocks)}")
    
    print(f"\n✅ 成功下載 {len(stock_data)} 支股票資料")
    return stock_data

def create_multi_stock_analysis(stock_data, zh_font):
    """建立多股票分析圖表 - 增強版"""
    if len(stock_data) == 0:
        return
    
    analysis_start = time.time()
    
    # 計算各股票的基本統計
    stats_data = []
    for stock_name, price_data in stock_data.items():
        # 計算更多統計指標
        returns = price_data.pct_change().dropna()
        annual_return = ((price_data.iloc[-1] / price_data.iloc[0]) ** (252/len(price_data)) - 1) * 100
        annual_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        max_drawdown = ((price_data / price_data.cummax()) - 1).min() * 100
        
        stats = {
            '股票': stock_name,
            '最高價': float(price_data.max()),
            '最低價': float(price_data.min()),
            '平均價': float(price_data.mean()),
            '標準差': float(price_data.std()),
            '總報酬率(%)': ((float(price_data.iloc[-1]) / float(price_data.iloc[0])) - 1) * 100,
            '年化報酬率(%)': annual_return,
            '年化波動率(%)': annual_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤(%)': max_drawdown,
            '資料點數': len(price_data)
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    print("\n📊 多股票進階統計分析:")
    from tabulate import tabulate
    print(tabulate(stats_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # 繪製多股票分析圖 - 擴展為6個子圖
    plt.figure(figsize=(20, 15))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(stock_data)))
    
    # 子圖1: 原始價格走勢
    plt.subplot(2, 3, 1)
    for i, (stock_name, price_data) in enumerate(stock_data.items()):
        plt.plot(price_data.index, price_data.values, 
                label=stock_name, linewidth=2, color=colors[i], alpha=0.8)
    
    plt.title('多股票價格走勢比較 (原始價格)', fontproperties=zh_font, fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.ylabel('價格', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.legend(prop=zh_font, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 子圖2: 標準化價格走勢
    plt.subplot(2, 3, 2)
    for i, (stock_name, price_data) in enumerate(stock_data.items()):
        normalized_price = (price_data / price_data.iloc[0]) * 100
        plt.plot(normalized_price.index, normalized_price.values, 
                label=stock_name, linewidth=2, color=colors[i], alpha=0.8)
    
    plt.title('多股票標準化走勢 (基準=100)', fontproperties=zh_font, fontsize=16, fontweight='bold')
    plt.xlabel('日期', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.ylabel('標準化價格', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.legend(prop=zh_font, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 子圖3: 年化報酬率比較
    plt.subplot(2, 3, 3)
    annual_returns = [stats['年化報酬率(%)'] for stats in stats_data]
    stock_names = [stats['股票'] for stats in stats_data]
    colors_bar = ['green' if r > 0 else 'red' for r in annual_returns]
    
    bars = plt.bar(range(len(stock_names)), annual_returns, color=colors_bar, alpha=0.7, edgecolor='black')
    plt.title('年化報酬率比較', fontproperties=zh_font, fontsize=16, fontweight='bold')
    plt.xlabel('股票', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.ylabel('年化報酬率 (%)', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.xticks(range(len(stock_names)), stock_names, rotation=45, fontproperties=zh_font, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在長條圖上顯示數值
    for bar, return_val in zip(bars, annual_returns):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                f'{return_val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # 子圖4: 風險-報酬散點圖
    plt.subplot(2, 3, 4)
    volatilities = [stats['年化波動率(%)'] for stats in stats_data]
    
    scatter = plt.scatter(volatilities, annual_returns, s=100, alpha=0.7, c=range(len(stock_names)), cmap='tab10')
    
    for i, name in enumerate(stock_names):
        plt.annotate(name, (volatilities[i], annual_returns[i]), xytext=(5, 5), 
                    textcoords='offset points', fontproperties=zh_font, fontsize=10)
    
    plt.title('風險-報酬分析', fontproperties=zh_font, fontsize=16, fontweight='bold')
    plt.xlabel('年化波動率 (%)', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.ylabel('年化報酬率 (%)', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 子圖5: 夏普比率比較
    plt.subplot(2, 3, 5)
    sharpe_ratios = [stats['夏普比率'] for stats in stats_data]
    colors_sharpe = ['green' if s > 0 else 'red' for s in sharpe_ratios]
    
    bars = plt.bar(range(len(stock_names)), sharpe_ratios, color=colors_sharpe, alpha=0.7, edgecolor='black')
    plt.title('夏普比率比較', fontproperties=zh_font, fontsize=16, fontweight='bold')
    plt.xlabel('股票', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.ylabel('夏普比率', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.xticks(range(len(stock_names)), stock_names, rotation=45, fontproperties=zh_font, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在長條圖上顯示數值
    for bar, sharpe_val in zip(bars, sharpe_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                f'{sharpe_val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # 子圖6: 最大回撤比較
    plt.subplot(2, 3, 6)
    max_drawdowns = [stats['最大回撤(%)'] for stats in stats_data]
    
    bars = plt.bar(range(len(stock_names)), max_drawdowns, color='red', alpha=0.7, edgecolor='black')
    plt.title('最大回撤比較', fontproperties=zh_font, fontsize=16, fontweight='bold')
    plt.xlabel('股票', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.ylabel('最大回撤 (%)', fontproperties=zh_font, fontsize=12, fontweight='bold')
    plt.xticks(range(len(stock_names)), stock_names, rotation=45, fontproperties=zh_font, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在長條圖上顯示數值
    for bar, dd_val in zip(bars, max_drawdowns):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height - 1,
                f'{dd_val:.1f}%', ha='center', va='top',
                fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    
    analysis_time = time.time() - analysis_start
    
    # 儲存多股票分析圖
    multi_stock_file = 'enhanced_multi_stock_analysis.png'
    plt.savefig(multi_stock_file, dpi=300, bbox_inches='tight')
    print(f"✅ 增強版多股票分析圖已儲存為: {multi_stock_file} (分析時間: {analysis_time:.2f}秒)")
    plt.show()
    
    # 額外的算力統計分析
    print(f"\n🧮 多股票分析算力統計:")
    print(f"分析時間: {analysis_time:.2f} 秒")
    print(f"處理股票數量: {len(stock_data)}")
    print(f"總資料點數: {sum(len(data) for data in stock_data.values()):,}")
    print(f"平均處理速度: {sum(len(data) for data in stock_data.values()) / analysis_time:.0f} 點/秒")

# 新增：詳細的演算法基準測試
def benchmark_algorithms(train_data, test_data, zh_font):
    """演算法基準測試 - 算力和精度比較"""
    print("\n🏁 開始演算法基準測試...")
    
    # 準備資料
    X_train, y_train = create_features(train_data, window_size=5)  # 使用較小窗口以節省時間
    X_test, y_test = create_features(test_data, window_size=5)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("❌ 資料不足，無法進行基準測試")
        return
    
    # 定義要測試的演算法
    algorithms = {
        '線性回歸': {
            'model': None,
            'train_func': lambda: train_linear_regression(X_train, y_train),
            'predict_func': None
        },
        '隨機森林': {
            'model': None,
            'train_func': lambda: train_random_forest(X_train, y_train),
            'predict_func': None
        },
        'XGBoost': {
            'model': None,
            'train_func': lambda: train_xgboost(X_train, y_train),
            'predict_func': None
        },
        '基因演算法': {
            'model': None,
            'train_func': lambda: train_genetic_algorithm(X_train, y_train),
            'predict_func': None
        }
    }
    
    results = []
    
    for algo_name, algo_config in algorithms.items():
        print(f"\n🔬 測試 {algo_name}...")
        
        # 建立資源監控器
        algo_monitor = ComputationalResourceMonitor()
        algo_monitor.start_monitoring()
        
        try:
            # 訓練階段
            train_start = time.time()
            model = algo_config['train_func']()
            train_time = time.time() - train_start
            
            # 預測階段
            pred_start = time.time()
            if algo_name == '基因演算法':
                predictions = model.predict(X_test)
            elif algo_name in ['線性回歸', '隨機森林', 'XGBoost']:
                predictions = model.predict(X_test)
            pred_time = time.time() - pred_start
            
            # 停止監控
            algo_monitor.stop_monitoring()
            stats = algo_monitor.get_statistics()
            
            # 計算誤差
            rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
            mae = np.mean(np.abs(y_test - predictions))
            
            # 記錄結果
            result = {
                '演算法': algo_name,
                '訓練時間(秒)': train_time,
                '預測時間(秒)': pred_time,
                '總時間(秒)': train_time + pred_time,
                'RMSE': rmse,
                'MAE': mae,
                '平均CPU(%)': stats.get('cpu_avg', 0),
                '最大CPU(%)': stats.get('cpu_max', 0),
                '平均記憶體(%)': stats.get('memory_avg', 0),
                '最大記憶體(%)': stats.get('memory_max', 0)
            }
            
            if stats.get('gpu_avg', 0) > 0:
                result['平均GPU(%)'] = stats.get('gpu_avg', 0)
                result['最大GPU(%)'] = stats.get('gpu_max', 0)
            
            results.append(result)
            print(f"✅ {algo_name} 完成 - 訓練: {train_time:.2f}s, 預測: {pred_time:.2f}s, RMSE: {rmse:.4f}")
            
        except Exception as e:
            algo_monitor.stop_monitoring()
            print(f"❌ {algo_name} 測試失敗: {e}")
            continue
    
    # 建立比較表格
    if results:
        results_df = pd.DataFrame(results)
        print("\n📊 演算法基準測試結果:")
        from tabulate import tabulate
        print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
        
        # 繪製基準測試圖表
        create_benchmark_charts(results_df, zh_font)
    
    return results

def train_linear_regression(X_train, y_train):
    """訓練線性回歸模型"""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """訓練隨機森林模型"""
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """訓練XGBoost模型"""
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_genetic_algorithm(X_train, y_train):
    """訓練基因演算法模型"""
    ga = GeneticAlgorithmPredictor(population_size=20, generations=30, mutation_rate=0.1)
    ga.evolve(X_train, y_train)
    return ga

def create_benchmark_charts(results_df, zh_font):
    """建立基準測試圖表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    algorithms = results_df['演算法'].values
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    # 子圖1: 執行時間比較
    total_times = results_df['總時間(秒)'].values
    bars1 = ax1.bar(algorithms, total_times, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('總執行時間比較', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax1.set_xlabel('演算法', fontproperties=zh_font, fontsize=12)
    ax1.set_ylabel('時間 (秒)', fontproperties=zh_font, fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 顯示數值
    for bar, time_val in zip(bars1, total_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子圖2: 預測精度比較 (RMSE)
    rmse_values = results_df['RMSE'].values
    bars2 = ax2.bar(algorithms, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('預測精度比較 (RMSE)', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax2.set_xlabel('演算法', fontproperties=zh_font, fontsize=12)
    ax2.set_ylabel('RMSE', fontproperties=zh_font, fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # 顯示數值
    for bar, rmse_val in zip(bars2, rmse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rmse_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子圖3: CPU使用率比較
    cpu_avg = results_df['平均CPU(%)'].values
    bars3 = ax3.bar(algorithms, cpu_avg, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('平均CPU使用率比較', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax3.set_xlabel('演算法', fontproperties=zh_font, fontsize=12)
    ax3.set_ylabel('CPU使用率 (%)', fontproperties=zh_font, fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 100)
    
    # 顯示數值
    for bar, cpu_val in zip(bars3, cpu_avg):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{cpu_val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 子圖4: 效率散點圖 (時間 vs 精度)
    ax4.scatter(total_times, rmse_values, s=100, alpha=0.7, c=range(len(algorithms)), cmap='Set3')
    
    for i, algo in enumerate(algorithms):
        ax4.annotate(algo, (total_times[i], rmse_values[i]), xytext=(5, 5), 
                    textcoords='offset points', fontproperties=zh_font, fontsize=10)
    
    ax4.set_title('效率分析 (時間 vs 精度)', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax4.set_xlabel('執行時間 (秒)', fontproperties=zh_font, fontsize=12)
    ax4.set_ylabel('RMSE', fontproperties=zh_font, fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    benchmark_file = 'algorithm_benchmark_results.png'
    plt.savefig(benchmark_file, dpi=300, bbox_inches='tight')
    print(f"✅ 基準測試結果圖已儲存為: {benchmark_file}")
    plt.show()

# 更新主函數以包含基準測試
def main_with_benchmark():
    """包含完整基準測試的主函數"""
    # 先執行原有的主函數
    main()
    
    # 如果有足夠的資料，執行基準測試
    try:
        print("\n" + "="*60)
        print("🏁 開始執行完整的演算法基準測試...")
        
        # 重新下載資料以確保一致性
        data, train, test, price_data = download_tsmc_data()
        if data is not None and len(train) > 50 and len(test) > 10:
            zh_font = setup_chinese_font()
            benchmark_results = benchmark_algorithms(train, test, zh_font)
            
            if benchmark_results:
                print("\n🎯 基準測試總結:")
                print("最快演算法:", min(benchmark_results, key=lambda x: x['總時間(秒)'])['演算法'])
                print("最準確演算法:", min(benchmark_results, key=lambda x: x['RMSE'])['演算法'])
                print("最省CPU演算法:", min(benchmark_results, key=lambda x: x['平均CPU(%)'])['演算法'])
                
                # 計算效率指標 (精度/時間)
                for result in benchmark_results:
                    efficiency = 1 / (result['RMSE'] * result['總時間(秒)'])
                    result['效率指標'] = efficiency
                
                best_efficiency = max(benchmark_results, key=lambda x: x['效率指標'])
                print("最高效率演算法:", best_efficiency['演算法'])
        else:
            print("⚠️ 資料不足，跳過基準測試")
            
    except Exception as e:
        print(f"❌ 基準測試執行失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 可以選擇執行基本版本或包含基準測試的完整版本
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        main_with_benchmark()
    else:
        main()