import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 参数设置 ==========
BASE_PATH = "../data/ODOC"
SOURCE_DOMAIN = "Domain1"
TARGET_DOMAIN = "Domain4"
FEDDG_DIR = "feddg_migration_all/migrated_images"
CYCLEGAN_DIR = "cyclegan_final_fixed/translated_images"
OUTPUT_DIR = "image_difference_analysis"
IMAGE_SIZE = 256

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 2. 自定义MSE、PSNR、SSIM计算函数 ==========
def calculate_mse(img1, img2):
    """计算均方误差 (Mean Squared Error)"""
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    """计算峰值信噪比 (Peak Signal-to-Noise Ratio)"""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim_simple(img1, img2):
    """简化的SSIM计算，不使用卷积，直接计算全局统计量"""
    # 常量
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # 计算全局统计量
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # SSIM计算公式
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return numerator / denominator

def calculate_ssim(img1, img2, window_size=7, k1=0.01, k2=0.03):
    """改进的SSIM计算，使用安全的卷积模式"""
    try:
        # 常量
        C1 = (k1 * 255) ** 2
        C2 = (k2 * 255) ** 2
        
        # 高斯窗口权重
        def gaussian_window(size, sigma=1.5):
            gauss = np.array([np.exp(-(x - size//2)**2 / (2*sigma**2)) 
                            for x in range(size)])
            gauss = gauss.reshape(-1, 1) * gauss.reshape(1, -1)
            return gauss / gauss.sum()
        
        # 创建窗口
        window = gaussian_window(window_size, sigma=1.5)
        
        # 确保图像是二维的
        if len(img1.shape) == 2:
            img1 = img1.reshape(img1.shape[0], img1.shape[1], 1)
            img2 = img2.reshape(img2.shape[0], img2.shape[1], 1)
        
        # 图像维度
        height, width, channels = img1.shape
        
        # 使用'mirror'或'reflect'模式代替'valid'
        mu1 = ndimage.convolve(img1, window[:, :, np.newaxis], mode='reflect')
        mu2 = ndimage.convolve(img2, window[:, :, np.newaxis], mode='reflect')
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ndimage.convolve(img1 ** 2, window[:, :, np.newaxis], mode='reflect') - mu1_sq
        sigma2_sq = ndimage.convolve(img2 ** 2, window[:, :, np.newaxis], mode='reflect') - mu2_sq
        sigma12 = ndimage.convolve(img1 * img2, window[:, :, np.newaxis], mode='reflect') - mu1_mu2
        
        # SSIM计算公式
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
        
        # 返回平均SSIM
        return np.mean(ssim_map)
        
    except Exception as e:
        print(f"SSIM计算失败，使用简化版本: {e}")
        return calculate_ssim_simple(img1, img2)

def calculate_image_metrics(img1_path, img2_path, img_name=""):
    """计算两个图像之间的MSE、PSNR、SSIM指标"""
    try:
        # 读取图像
        img1 = Image.open(img1_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
        img2 = Image.open(img2_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # 转换为numpy数组
        img1_arr = np.array(img1, dtype=np.float32)
        img2_arr = np.array(img2, dtype=np.float32)
        
        # 计算三个核心指标
        metrics = {}
        metrics['mse'] = calculate_mse(img1_arr, img2_arr)
        metrics['psnr'] = calculate_psnr(img1_arr, img2_arr)
        metrics['ssim'] = calculate_ssim(img1_arr, img2_arr)
        
        return metrics, img1_arr, img2_arr
        
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        return None, None, None

# ========== 3. 检查文件存在的辅助函数 ==========
def check_and_get_image_paths(source_path, img_name):
    """检查并获取图像路径"""
    # FedDG迁移图像路径
    feddg_img_name = f"migrated_{img_name}"
    feddg_img_path = os.path.join(FEDDG_DIR, feddg_img_name)
    
    # CycleGAN迁移图像路径
    cyclegan_img_name = f"cyclegan_{img_name}"
    cyclegan_img_path = os.path.join(CYCLEGAN_DIR, cyclegan_img_name)
    
    source_img_path = os.path.join(source_path, img_name)
    
    # 检查所有文件是否存在
    missing_files = []
    if not os.path.exists(source_img_path):
        missing_files.append(f"Source: {source_img_path}")
    if not os.path.exists(feddg_img_path):
        missing_files.append(f"FedDG: {feddg_img_path}")
    if not os.path.exists(cyclegan_img_path):
        missing_files.append(f"CycleGAN: {cyclegan_img_path}")
    
    if missing_files:
        return None, None, None, missing_files
    
    return source_img_path, feddg_img_path, cyclegan_img_path, []

# ========== 4. 加载所有图像并计算指标 ==========
def analyze_all_images():
    """分析所有图像的差异"""
    print("="*70)
    print("开始分析迁移图像差异...")
    print("="*70)
    
    # 获取源域图像
    source_path = os.path.join(BASE_PATH, SOURCE_DOMAIN, "train", "imgs")
    
    # 检查源路径是否存在
    if not os.path.exists(source_path):
        print(f"错误：源路径不存在: {source_path}")
        return None, None
    
    source_images = sorted([f for f in os.listdir(source_path) if f.lower().endswith('.png')])
    
    # 检查图像数量
    if len(source_images) == 0:
        print("错误：源路径中没有找到PNG图像")
        return None, None
    
    # 限制样本数量用于分析
    num_samples = min(20, len(source_images))
    source_images = source_images[:num_samples]
    
    print(f"找到 {len(source_images)} 个源图像")
    print(f"分析 {num_samples} 个样本...")
    
    # 存储所有指标
    all_metrics = {
        'feddg': [],
        'cyclegan': [],
        'feddg_raw': {},
        'cyclegan_raw': {}
    }
    
    sample_names = []
    processed_count = 0
    
    for i, img_name in enumerate(source_images):
        print(f"\n处理样本 {i+1}/{num_samples}: {img_name}")
        
        # 获取图像路径
        source_img_path, feddg_img_path, cyclegan_img_path, missing = check_and_get_image_paths(source_path, img_name)
        
        if missing:
            print(f"  缺少文件: {', '.join(missing)}")
            continue
        
        # 计算FedDG差异指标
        print(f"  计算FedDG指标...")
        feddg_metrics, source_img, feddg_img = calculate_image_metrics(
            source_img_path, feddg_img_path, f"FedDG_{img_name}"
        )
        
        # 计算CycleGAN差异指标
        print(f"  计算CycleGAN指标...")
        cyclegan_metrics, _, cyclegan_img = calculate_image_metrics(
            source_img_path, cyclegan_img_path, f"CycleGAN_{img_name}"
        )
        
        if feddg_metrics and cyclegan_metrics:
            sample_names.append(img_name)
            all_metrics['feddg'].append(feddg_metrics)
            all_metrics['cyclegan'].append(cyclegan_metrics)
            
            # 存储原始数据用于详细分析
            all_metrics['feddg_raw'][img_name] = {
                'metrics': feddg_metrics,
                'source': source_img,
                'transferred': feddg_img
            }
            all_metrics['cyclegan_raw'][img_name] = {
                'metrics': cyclegan_metrics,
                'source': source_img,
                'transferred': cyclegan_img
            }
            
            processed_count += 1
            
            # 打印关键指标
            print(f"  FedDG - PSNR: {feddg_metrics['psnr']:.2f} dB, "
                  f"SSIM: {feddg_metrics['ssim']:.3f}, "
                  f"MSE: {feddg_metrics['mse']:.1f}")
            print(f"  CycleGAN - PSNR: {cyclegan_metrics['psnr']:.2f} dB, "
                  f"SSIM: {cyclegan_metrics['ssim']:.3f}, "
                  f"MSE: {cyclegan_metrics['mse']:.1f}")
        else:
            print(f"  指标计算失败")
    
    if processed_count > 0:
        print(f"\n成功分析 {processed_count}/{num_samples} 个样本")
        return all_metrics, sample_names
    else:
        print("没有成功分析的样本")
        # 检查可能的原因
        print("\n检查可能的问题:")
        print(f"1. 检查FEDDG_DIR: {FEDDG_DIR}")
        print(f"2. 检查CYCLEGAN_DIR: {CYCLEGAN_DIR}")
        print(f"3. 检查源路径: {source_path}")
        return None, None

# ========== 5. 简化版的可视化函数 ==========
def create_simple_comparison(all_metrics, sample_names):
    """创建简化的指标对比可视化"""
    print("\n生成指标对比可视化图表...")
    
    # 提取指标数据
    feddg_data = all_metrics['feddg']
    cyclegan_data = all_metrics['cyclegan']
    
    # 计算平均指标
    feddg_means = {
        'mse': np.mean([d['mse'] for d in feddg_data]),
        'psnr': np.mean([d['psnr'] for d in feddg_data]),
        'ssim': np.mean([d['ssim'] for d in feddg_data])
    }
    cyclegan_means = {
        'mse': np.mean([d['mse'] for d in cyclegan_data]),
        'psnr': np.mean([d['psnr'] for d in cyclegan_data]),
        'ssim': np.mean([d['ssim'] for d in cyclegan_data])
    }
    
    # 1. 创建简单的柱状图对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_list = ['mse', 'psnr', 'ssim']
    metric_labels = ['MSE (越低越好)', 'PSNR [dB] (越高越好)', 'SSIM (越高越好)']
    
    for idx, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
        ax = axes[idx]
        
        # 设置位置
        x = [0, 1]
        methods = ['FedDG', 'CycleGAN']
        values = [feddg_means[metric], cyclegan_means[metric]]
        colors = ['blue', 'red']
        
        # 绘制柱状图
        bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.set_xlabel('Method')
        ax.set_ylabel(label)
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparison of Image Quality Metrics: FedDG vs CycleGAN', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), 
                dpi=150, bbox_inches='tight')
    
    # 2. 创建折线图展示样本间差异
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 提取所有样本的指标
    feddg_psnr = [d['psnr'] for d in feddg_data]
    feddg_ssim = [d['ssim'] for d in feddg_data]
    feddg_mse = [d['mse'] for d in feddg_data]
    
    cyclegan_psnr = [d['psnr'] for d in cyclegan_data]
    cyclegan_ssim = [d['ssim'] for d in cyclegan_data]
    cyclegan_mse = [d['mse'] for d in cyclegan_data]
    
    # 生成样本索引
    x = range(len(sample_names))
    
    # PSNR趋势
    axes[0].plot(x, feddg_psnr, 'o-', label='FedDG', linewidth=2, markersize=6, color='blue')
    axes[0].plot(x, cyclegan_psnr, 's-', label='CycleGAN', linewidth=2, markersize=6, color='red')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_title('PSNR Across Different Samples')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM趋势
    axes[1].plot(x, feddg_ssim, 'o-', label='FedDG', linewidth=2, markersize=6, color='blue')
    axes[1].plot(x, cyclegan_ssim, 's-', label='CycleGAN', linewidth=2, markersize=6, color='red')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Across Different Samples')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MSE趋势
    axes[2].plot(x, feddg_mse, 'o-', label='FedDG', linewidth=2, markersize=6, color='blue')
    axes[2].plot(x, cyclegan_mse, 's-', label='CycleGAN', linewidth=2, markersize=6, color='red')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('MSE Across Different Samples')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Performance Trends Across Different Image Samples', 
                 fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_trends.png'), 
                dpi=150, bbox_inches='tight')
    
    # 3. 创建总结对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算改进百分比
    improvement = {}
    improvement['MSE'] = (feddg_means['mse'] - cyclegan_means['mse']) / feddg_means['mse'] * 100
    improvement['PSNR'] = (cyclegan_means['psnr'] - feddg_means['psnr']) / feddg_means['psnr'] * 100
    improvement['SSIM'] = (cyclegan_means['ssim'] - feddg_means['ssim']) / feddg_means['ssim'] * 100
    
    # 绘制改进百分比
    metrics_names = ['MSE', 'PSNR', 'SSIM']
    values = [improvement['MSE'], improvement['PSNR'], improvement['SSIM']]
    colors = ['green' if (name == 'MSE' and x > 0) or (name != 'MSE' and x > 0) else 'red' 
              for name, x in zip(metrics_names, values)]
    
    bars = ax.bar(metrics_names, values, color=colors, alpha=0.7)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:+.1f}%', ha='center', va='bottom' if value > 0 else 'top')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Improvement Percentage (%)')
    ax.set_title('CycleGAN Improvement Over FedDG (正数表示CycleGAN更好)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'improvement_comparison.png'), 
                dpi=150, bbox_inches='tight')
    
    plt.close('all')
    print("可视化图表已生成")

# ========== 6. 生成统计报告 ==========
def generate_statistical_report(all_metrics, sample_names):
    """生成统计报告"""
    print("\n生成统计报告...")
    
    report_path = os.path.join(OUTPUT_DIR, 'image_difference_report.txt')
    
    feddg_data = all_metrics['feddg']
    cyclegan_data = all_metrics['cyclegan']
    
    with open(report_path, 'w') as f:
        f.write("IMAGE TRANSFER DIFFERENCE ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Number of Samples: {len(sample_names)}\n")
        f.write(f"Source Domain: {SOURCE_DOMAIN}\n")
        f.write(f"Target Domain: {TARGET_DOMAIN}\n\n")
        
        f.write("1. AVERAGE METRICS COMPARISON\n")
        f.write("-"*40 + "\n")
        f.write("Metric          FedDG (Mean)     CycleGAN (Mean)    Difference\n")
        f.write("----------------------------------------------------------------\n")
        
        metrics = ['mse', 'psnr', 'ssim']
        metric_names = ['MSE', 'PSNR [dB]', 'SSIM']
        
        for metric, name in zip(metrics, metric_names):
            feddg_mean = np.mean([d[metric] for d in feddg_data])
            cyclegan_mean = np.mean([d[metric] for d in cyclegan_data])
            
            if metric == 'mse':
                diff = feddg_mean - cyclegan_mean  # MSE越低越好
                better = "FedDG" if diff > 0 else "CycleGAN"
            else:
                diff = cyclegan_mean - feddg_mean  # PSNR和SSIM越高越好
                better = "CycleGAN" if diff > 0 else "FedDG"
            
            diff_pct = (diff / feddg_mean * 100) if feddg_mean != 0 else 0
            
            f.write(f"{name:<15} {feddg_mean:>12.2f} {cyclegan_mean:>17.2f} "
                   f"{diff:>+11.2f} ({better})\n")
        
        f.write("\n2. METHOD RECOMMENDATION\n")
        f.write("-"*40 + "\n")
        
        # 分析哪个方法更好
        feddg_better = 0
        cyclegan_better = 0
        
        for metric in metrics:
            feddg_mean = np.mean([d[metric] for d in feddg_data])
            cyclegan_mean = np.mean([d[metric] for d in cyclegan_data])
            
            if metric == 'mse':
                if feddg_mean < cyclegan_mean:
                    feddg_better += 1
                else:
                    cyclegan_better += 1
            else:
                if feddg_mean > cyclegan_mean:
                    feddg_better += 1
                else:
                    cyclegan_better += 1
        
        f.write(f"Metrics where FedDG is better: {feddg_better}/3\n")
        f.write(f"Metrics where CycleGAN is better: {cyclegan_better}/3\n\n")
        
        if feddg_better > cyclegan_better:
            f.write("RECOMMENDATION: FedDG is the preferred method.\n")
            f.write("Reason: FedDG performs better on more quality metrics.\n")
        elif cyclegan_better > feddg_better:
            f.write("RECOMMENDATION: CycleGAN is the preferred method.\n")
            f.write("Reason: CycleGAN performs better on more quality metrics.\n")
        else:
            f.write("RECOMMENDATION: Both methods have comparable performance.\n")
            f.write("Choice should be based on specific requirements.\n")
        
        f.write("\n3. GENERATED VISUALIZATIONS\n")
        f.write("-"*40 + "\n")
        f.write("The following analysis files have been generated:\n")
        for filename in sorted(os.listdir(OUTPUT_DIR)):
            if filename.endswith('.png'):
                f.write(f"• {filename}\n")
    
    print(f"统计报告已保存至: {report_path}")

# ========== 7. 主函数 ==========
def main():
    """主函数"""
    print("="*70)
    print("IMAGE TRANSFER DIFFERENCE ANALYSIS")
    print("="*70)
    print("Metrics: MSE, PSNR, SSIM")
    print("Methods: FedDG vs CycleGAN")
    print("="*70)
    
    # 检查目录是否存在
    print("\n检查目录...")
    print(f"FedDG目录: {FEDDG_DIR} - {'存在' if os.path.exists(FEDDG_DIR) else '不存在'}")
    print(f"CycleGAN目录: {CYCLEGAN_DIR} - {'存在' if os.path.exists(CYCLEGAN_DIR) else '不存在'}")
    
    if not os.path.exists(FEDDG_DIR):
        print(f"错误：FedDG目录不存在: {FEDDG_DIR}")
        print("请先运行FedDG代码生成迁移图像")
        return
    
    if not os.path.exists(CYCLEGAN_DIR):
        print(f"错误：CycleGAN目录不存在: {CYCLEGAN_DIR}")
        print("请先运行CycleGAN代码生成迁移图像")
        return
    
    # 1. 分析所有图像
    all_metrics, sample_names = analyze_all_images()
    
    if all_metrics is None:
        print("\n分析失败：没有找到足够的图像数据")
        print("请检查：")
        print(f"1. 确保 {FEDDG_DIR} 中有迁移图像")
        print(f"2. 确保 {CYCLEGAN_DIR} 中有迁移图像")
        print(f"3. 确保源路径 {os.path.join(BASE_PATH, SOURCE_DOMAIN, 'train', 'imgs')} 中有图像")
        return
    
    # 2. 创建可视化
    create_simple_comparison(all_metrics, sample_names)
    
    # 3. 生成统计报告
    generate_statistical_report(all_metrics, sample_names)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n所有结果已保存至目录: {os.path.abspath(OUTPUT_DIR)}")
    print("\n生成的文件:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        file_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / 1024
            if filename.endswith('.png'):
                print(f"  📊 {filename} ({file_size:.1f} KB)")
            elif filename.endswith('.txt'):
                print(f"  📄 {filename} ({file_size:.1f} KB)")

# ========== 8. 运行程序 ==========
if __name__ == "__main__":
    main()
    