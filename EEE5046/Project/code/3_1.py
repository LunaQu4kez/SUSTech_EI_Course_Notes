import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 参数设置 ==========
BASE_PATH = "../data/ODOC"
SOURCE_DOMAIN = "Domain1"
TARGET_DOMAIN = "Domain4"  # Task1中Dice最低的域
NUM_VISUALIZATION_SAMPLES = 5  # 只选择5个样本进行可视化对比
OUTPUT_DIR = "feddg_migration_all"
MIGRATED_DIR = os.path.join(OUTPUT_DIR, "migrated_images")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MIGRATED_DIR, exist_ok=True)

# ========== 2. FedDG方法核心函数 ==========
def feddg_style_transfer(source_img_path, target_img_path, lambda_param=0.5, mask_ratio=0.5):
    """
    使用FedDG方法进行风格迁移
    核心思想：在频域混合幅度谱，保持源图像的相位谱
    
    参数：
    - source_img_path: 源图像路径（Domain1）
    - target_img_path: 目标图像路径（Domain4）
    - lambda_param: 插值比例 (0-1)，控制风格迁移强度
    - mask_ratio: 掩码比例，控制混合区域大小
    """
    try:
        # 1. 读取源图像和目标图像
        source_img = Image.open(source_img_path).convert('L')
        target_img = Image.open(target_img_path).convert('L')
        
        # 确保图像尺寸相同
        if source_img.size != target_img.size:
            # 调整目标图像大小以匹配源图像
            target_img = target_img.resize(source_img.size, Image.Resampling.LANCZOS)
        
        # 转换为numpy数组
        source_array = np.array(source_img, dtype=np.float32)
        target_array = np.array(target_img, dtype=np.float32)
        
        # 2. 进行2D傅里叶变换
        source_f = np.fft.fft2(source_array)
        target_f = np.fft.fft2(target_array)
        
        # 3. 频谱中心化
        source_fshift = np.fft.fftshift(source_f)
        target_fshift = np.fft.fftshift(target_f)
        
        # 4. 提取幅度谱和相位谱
        source_amplitude = np.abs(source_fshift)
        source_phase = np.angle(source_fshift)
        
        target_amplitude = np.abs(target_fshift)
        target_phase = np.angle(target_fshift)
        
        # 5. 创建混合掩码（类似FedDG论文中的方法）
        height, width = source_amplitude.shape
        center_y, center_x = height // 2, width // 2
        
        # 创建距离矩阵
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # 创建掩码：中心区域为1（低频），边缘区域为0（高频）
        max_radius = min(center_x, center_y)
        mask_radius = int(mask_ratio * max_radius)
        
        # 低频掩码（中心区域）
        low_freq_mask = dist_from_center <= mask_radius
        
        # 高频掩码（边缘区域）
        high_freq_mask = dist_from_center > mask_radius
        
        # 6. FedDG幅度谱混合公式
        # A_{i,λ}^{k-n} = (1-λ)A_i^k * (1-M) + λ A_j^n * M
        # 其中：A_i^k是源幅度谱，A_j^n是目标幅度谱
        # M是掩码，λ是插值比例
        
        # 初始化混合幅度谱
        mixed_amplitude = np.zeros_like(source_amplitude)
        
        # 应用FedDG公式
        # 对于低频区域：主要使用目标域的风格（Domain4）
        mixed_amplitude[low_freq_mask] = (
            (1 - lambda_param) * source_amplitude[low_freq_mask] * (1 - 1) +  # (1-M)部分
            lambda_param * target_amplitude[low_freq_mask] * 1  # M部分
        )
        
        # 对于高频区域：主要保持源域的内容（Domain1）
        mixed_amplitude[high_freq_mask] = (
            (1 - lambda_param) * source_amplitude[high_freq_mask] * 1 +  # (1-M)部分
            lambda_param * target_amplitude[high_freq_mask] * (1 - 1)   # M部分
        )
        
        # 7. 重建图像：使用混合的幅度谱 + 源图像的相位谱
        mixed_fshift = mixed_amplitude * np.exp(1j * source_phase)
        
        # 8. 逆中心化
        mixed_f = np.fft.ifftshift(mixed_fshift)
        
        # 9. 逆傅里叶变换
        mixed_array = np.abs(np.fft.ifft2(mixed_f))
        
        # 10. 确保值在合理范围内
        mixed_array = np.clip(mixed_array, 0, 255)
        
        return {
            'source': source_array,
            'target': target_array,
            'mixed': mixed_array,
            'source_amplitude': source_amplitude,
            'target_amplitude': target_amplitude,
            'mixed_amplitude': mixed_amplitude,
            'source_phase': source_phase,
            'low_freq_mask': low_freq_mask,
            'high_freq_mask': high_freq_mask,
            'lambda_param': lambda_param,
            'mask_ratio': mask_ratio,
            'source_name': os.path.basename(source_img_path),
            'target_name': os.path.basename(target_img_path)
        }
        
    except Exception as e:
        print(f"Error in FedDG transfer: {e}")
        return None

# ========== 3. 参数搜索函数 ==========
def find_best_parameters(source_img_path, target_img_path):
    """
    搜索最佳参数组合
    """
    print("Searching for optimal parameters...")
    
    best_result = None
    best_params = {'lambda': 0.5, 'mask_ratio': 0.5}
    best_score = -1
    
    # 参数网格搜索
    lambda_values = [0.3, 0.5, 0.7]
    mask_ratios = [0.3, 0.5, 0.7]
    
    for lambda_val in lambda_values:
        for mask_ratio in mask_ratios:
            result = feddg_style_transfer(source_img_path, target_img_path, 
                                         lambda_param=lambda_val, mask_ratio=mask_ratio)
            
            if result is None:
                continue
            
            # 计算迁移质量评分（基于幅度谱相似性）
            # 计算源图像和混合图像的相关系数
            source_corr = np.corrcoef(result['source'].flatten(), result['mixed'].flatten())[0, 1]
            
            # 计算目标图像和混合图像的相关系数
            target_corr = np.corrcoef(result['target'].flatten(), result['mixed'].flatten())[0, 1]
            
            # 综合评分：希望在保持源内容的同时获得目标风格
            score = 0.6 * source_corr + 0.4 * target_corr
            
            if score > best_score:
                best_score = score
                best_params = {'lambda': lambda_val, 'mask_ratio': mask_ratio}
                best_result = result
    
    print(f"Best parameters: λ={best_params['lambda']:.1f}, mask_ratio={best_params['mask_ratio']:.1f}")
    print(f"Best score: {best_score:.3f}")
    
    return best_params

# ========== 4. 批量处理所有图像 ==========
def process_all_images(source_domain_path, target_domain_path, best_params):
    """
    处理所有Domain1图像，迁移到Domain4风格
    """
    print("\n" + "="*70)
    print("Processing ALL Domain1 images...")
    print("="*70)
    
    # 获取所有图像
    source_images = sorted([f for f in os.listdir(source_domain_path) if f.lower().endswith('.png')])
    target_images = sorted([f for f in os.listdir(target_domain_path) if f.lower().endswith('.png')])
    
    if not source_images or not target_images:
        print("Error: No images found in one or both domains")
        return [], []
    
    print(f"Found {len(source_images)} source images and {len(target_images)} target images")
    
    # 存储结果
    all_mixed_images = []
    all_source_names = []
    processed_count = 0
    
    # 为每个源图像选择一个目标图像（循环使用）
    for i, source_img_name in enumerate(source_images):
        # 选择目标图像（循环使用目标图像集）
        target_idx = i % len(target_images)
        target_img_name = target_images[target_idx]
        
        source_img_path = os.path.join(source_domain_path, source_img_name)
        target_img_path = os.path.join(target_domain_path, target_img_name)
        
        # 进行迁移
        result = feddg_style_transfer(
            source_img_path, target_img_path,
            lambda_param=best_params['lambda'],
            mask_ratio=best_params['mask_ratio']
        )
        
        if result is not None:
            # 保存迁移后的图像
            mixed_img = Image.fromarray(result['mixed'].astype(np.uint8))
            mixed_save_path = os.path.join(MIGRATED_DIR, f"migrated_{source_img_name}")
            mixed_img.save(mixed_save_path)
            
            all_mixed_images.append(result)
            all_source_names.append(source_img_name)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(source_images)} images...")
        else:
            print(f"  Failed to process: {source_img_name}")
    
    print(f"\nSuccessfully processed {processed_count}/{len(source_images)} images")
    return all_mixed_images, all_source_names

# ========== 5. 选择代表性样本进行可视化 ==========
def select_representative_samples(all_results, all_names, num_samples=5):
    """
    从所有结果中选择代表性样本进行可视化
    选择策略：基于图像多样性（直方图分布）
    """
    if len(all_results) <= num_samples:
        return all_results, all_names
    
    print(f"\nSelecting {num_samples} representative samples from {len(all_results)} results...")
    
    # 计算每个混合图像的直方图特征
    hist_features = []
    for result in all_results:
        hist, _ = np.histogram(result['mixed'].flatten(), bins=50, range=(0, 255))
        hist_features.append(hist / np.sum(hist))  # 归一化
    
    # 使用k-means聚类选择多样本
    from sklearn.cluster import KMeans
    hist_features_array = np.array(hist_features)
    
    # 执行k-means聚类
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(hist_features_array)
    
    # 从每个簇中选择一个代表性样本（最接近簇中心）
    selected_indices = []
    for cluster_id in range(num_samples):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            # 计算每个样本到簇中心的距离
            distances = []
            for idx in cluster_indices:
                distance = np.linalg.norm(hist_features_array[idx] - kmeans.cluster_centers_[cluster_id])
                distances.append(distance)
            
            # 选择距离最小的样本
            best_in_cluster = cluster_indices[np.argmin(distances)]
            selected_indices.append(best_in_cluster)
    
    # 如果簇数不够，补充随机样本
    if len(selected_indices) < num_samples:
        all_indices = list(range(len(all_results)))
        remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
        additional_needed = num_samples - len(selected_indices)
        
        if len(remaining_indices) >= additional_needed:
            import random
            random.seed(42)
            selected_indices.extend(random.sample(remaining_indices, additional_needed))
    
    # 确保不超过num_samples
    selected_indices = selected_indices[:num_samples]
    
    selected_results = [all_results[i] for i in selected_indices]
    selected_names = [all_names[i] for i in selected_indices]
    
    print(f"Selected samples: {selected_names}")
    return selected_results, selected_names

# ========== 6. 可视化函数 ==========
def visualize_feddg_results(result, domain_name, save_path):
    """
    可视化FedDG迁移结果
    """
    fig = plt.figure(figsize=(24, 16))  # 增大画布尺寸
    
    # 创建子图布局 - 增加hspace和wspace来增大间距
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.5)  # 增加间距
    
    # 第1行：原始图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result['source'], cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f"Source: {SOURCE_DOMAIN}\n{result['source_name']}", fontsize=14, pad=15)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(result['target'], cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f"Target: {TARGET_DOMAIN}\n{result['target_name']}", fontsize=14, pad=15)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(result['mixed'], cmap='gray', vmin=0, vmax=255)
    ax3.set_title(f"FedDG Result\nλ={result['lambda_param']:.1f}, mask={result['mask_ratio']:.1f}", 
                  fontsize=14, fontweight='bold', pad=15)
    ax3.axis('off')
    
    # 第1行第4个位置：空出来增加间距
    ax_empty1 = fig.add_subplot(gs[0, 3])
    ax_empty1.axis('off')
    
    # 第2行：幅度谱
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(np.log1p(result['source_amplitude']), cmap='hot')
    ax4.set_title(f"Source Amplitude Spectrum (log scale)", fontsize=14, pad=15)
    ax4.axis('off')
    # 增加颜色条与图像的距离
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.08)
    cbar4.ax.tick_params(labelsize=10)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(np.log1p(result['target_amplitude']), cmap='hot')
    ax5.set_title(f"Target Amplitude Spectrum (log scale)", fontsize=14, pad=15)
    ax5.axis('off')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.08)
    cbar5.ax.tick_params(labelsize=10)
    
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(np.log1p(result['mixed_amplitude']), cmap='hot')
    ax6.set_title(f"Mixed Amplitude Spectrum (log scale)", fontsize=14, pad=15)
    ax6.axis('off')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.08)
    cbar6.ax.tick_params(labelsize=10)
    
    # 掩码可视化
    ax7 = fig.add_subplot(gs[1, 3])
    mask_display = np.zeros_like(result['low_freq_mask'], dtype=np.float32)
    mask_display[result['low_freq_mask']] = 1.0  # 低频区域
    mask_display[result['high_freq_mask']] = 0.5  # 高频区域
    im7 = ax7.imshow(mask_display, cmap='coolwarm')
    ax7.set_title(f"Frequency Mask\nLow frequency = 1.0\nHigh frequency = 0.5", 
                  fontsize=14, pad=15)
    ax7.axis('off')
    cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.08)
    cbar7.ax.tick_params(labelsize=10)
    
    # 第3行：相位谱和差异
    ax8 = fig.add_subplot(gs[2, 0])
    im8 = ax8.imshow(result['source_phase'], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax8.set_title(f"Source Phase Spectrum", fontsize=14, pad=15)
    ax8.axis('off')
    cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.08)
    cbar8.ax.tick_params(labelsize=10)
    
    # 差异图像 - Source vs Mixed
    ax9 = fig.add_subplot(gs[2, 1])
    diff_source_mixed = np.abs(result['source'] - result['mixed'])
    im9 = ax9.imshow(diff_source_mixed, cmap='hot')
    ax9.set_title(f"Difference: Source vs Mixed\nMSE: {np.mean(diff_source_mixed**2):.1f}", 
                  fontsize=14, pad=15)
    ax9.axis('off')
    cbar9 = plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.08)
    cbar9.ax.tick_params(labelsize=10)
    
    # 差异图像 - Target vs Mixed
    ax10 = fig.add_subplot(gs[2, 2])
    diff_target_mixed = np.abs(result['target'] - result['mixed'])
    im10 = ax10.imshow(diff_target_mixed, cmap='hot')
    ax10.set_title(f"Difference: Target vs Mixed\nMSE: {np.mean(diff_target_mixed**2):.1f}", 
                   fontsize=14, pad=15)
    ax10.axis('off')
    cbar10 = plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.08)
    cbar10.ax.tick_params(labelsize=10)
    
    # 直方图比较
    ax11 = fig.add_subplot(gs[2, 3])
    ax11.hist(result['source'].flatten(), bins=50, alpha=0.7, label='Source', 
              color='blue', edgecolor='black', linewidth=0.5)
    ax11.hist(result['target'].flatten(), bins=50, alpha=0.7, label='Target', 
              color='red', edgecolor='black', linewidth=0.5)
    ax11.hist(result['mixed'].flatten(), bins=50, alpha=0.7, label='Mixed', 
              color='green', edgecolor='black', linewidth=0.5)
    ax11.set_title("Intensity Histograms Comparison", fontsize=14, pad=15)
    ax11.set_xlabel("Pixel Intensity (0-255)", fontsize=12)
    ax11.set_ylabel("Frequency", fontsize=12)
    ax11.legend(fontsize=11, loc='upper right')
    ax11.grid(True, alpha=0.3, linestyle='--')
    ax11.tick_params(axis='both', which='major', labelsize=10)
    
    # 添加整体标题
    plt.suptitle(f"FedDG Style Transfer: {SOURCE_DOMAIN} → {TARGET_DOMAIN}\n"
                 f"Sample: {domain_name} | "
                 f"Correlation (Source-Mixed): {np.corrcoef(result['source'].flatten(), result['mixed'].flatten())[0, 1]:.3f} | "
                 f"Correlation (Target-Mixed): {np.corrcoef(result['target'].flatten(), result['mixed'].flatten())[0, 1]:.3f}", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 调整布局，增加边界
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.5)  # 增加pad_inches
    plt.close()

# ========== 7. 创建对比可视化 ==========
def create_comparison_visualization(selected_results, selected_names):
    """
    创建所有选定样本的对比可视化
    """
    num_samples = len(selected_results)
    
    # 创建3行对比图
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i, (result, name) in enumerate(zip(selected_results, selected_names)):
        # 第1行：源图像
        axes[0, i].imshow(result['source'], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f"Source: {name}", fontsize=10)
        axes[0, i].axis('off')
        
        # 第2行：目标图像
        axes[1, i].imshow(result['target'], cmap='gray', vmin=0, vmax=255)
        axes[1, i].set_title(f"Target: {result['target_name']}", fontsize=10)
        axes[1, i].axis('off')
        
        # 第3行：混合图像
        axes[2, i].imshow(result['mixed'], cmap='gray', vmin=0, vmax=255)
        axes[2, i].set_title(f"Mixed Result\nλ={result['lambda_param']:.1f}", fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_overview.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison visualization saved to: {comparison_path}")
    
    # 创建统计汇总图
    create_statistics_summary(selected_results)

# ========== 8. 创建统计汇总 ==========
def create_statistics_summary(selected_results):
    """
    创建统计汇总图
    """
    num_samples = len(selected_results)
    
    # 计算统计指标
    metrics_list = []
    for result in selected_results:
        # 计算基本指标
        mse_source_mixed = np.mean((result['source'] - result['mixed'])**2)
        mse_target_mixed = np.mean((result['target'] - result['mixed'])**2)
        
        max_pixel = 255.0
        psnr_source_mixed = 10 * np.log10(max_pixel**2 / mse_source_mixed) if mse_source_mixed > 0 else float('inf')
        psnr_target_mixed = 10 * np.log10(max_pixel**2 / mse_target_mixed) if mse_target_mixed > 0 else float('inf')
        
        corr_source_mixed = np.corrcoef(result['source'].flatten(), result['mixed'].flatten())[0, 1]
        corr_target_mixed = np.corrcoef(result['target'].flatten(), result['mixed'].flatten())[0, 1]
        
        metrics_list.append({
            'name': result['source_name'],
            'mse_source': mse_source_mixed,
            'mse_target': mse_target_mixed,
            'psnr_source': psnr_source_mixed,
            'psnr_target': psnr_target_mixed,
            'corr_source': corr_source_mixed,
            'corr_target': corr_target_mixed
        })
    
    # 创建统计图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 相关系数对比
    names = [m['name'] for m in metrics_list]
    corr_source = [m['corr_source'] for m in metrics_list]
    corr_target = [m['corr_target'] for m in metrics_list]
    
    x = range(num_samples)
    axes[0, 0].bar(x, corr_source, width=0.4, label='Source-Mixed', color='blue', alpha=0.7)
    axes[0, 0].bar([i + 0.4 for i in x], corr_target, width=0.4, label='Target-Mixed', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Correlation Coefficient')
    axes[0, 0].set_title('Correlation with Source and Target')
    axes[0, 0].set_xticks([i + 0.2 for i in x])
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PSNR对比
    psnr_source = [m['psnr_source'] for m in metrics_list]
    psnr_target = [m['psnr_target'] for m in metrics_list]
    
    axes[0, 1].bar(x, psnr_source, width=0.4, label='Source-Mixed', color='blue', alpha=0.7)
    axes[0, 1].bar([i + 0.4 for i in x], psnr_target, width=0.4, label='Target-Mixed', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
    axes[0, 1].set_xticks([i + 0.2 for i in x])
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MSE对比
    mse_source = [m['mse_source'] for m in metrics_list]
    mse_target = [m['mse_target'] for m in metrics_list]
    
    axes[1, 0].bar(x, mse_source, width=0.4, label='Source-Mixed', color='blue', alpha=0.7)
    axes[1, 0].bar([i + 0.4 for i in x], mse_target, width=0.4, label='Target-Mixed', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Mean Squared Error')
    axes[1, 0].set_xticks([i + 0.2 for i in x])
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 参数对比
    lambda_values = [r['lambda_param'] for r in selected_results]
    mask_ratios = [r['mask_ratio'] for r in selected_results]
    
    axes[1, 1].bar(x, lambda_values, width=0.4, label='Lambda (λ)', color='green', alpha=0.7)
    axes[1, 1].bar([i + 0.4 for i in x], mask_ratios, width=0.4, label='Mask Ratio', color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].set_title('FedDG Parameters')
    axes[1, 1].set_xticks([i + 0.2 for i in x])
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'FedDG Performance Statistics for Selected Samples', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    stats_path = os.path.join(OUTPUT_DIR, "performance_statistics.png")
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Performance statistics saved to: {stats_path}")
    
    return metrics_list

# ========== 9. 主处理流程 ==========
print("="*70)
print("FedDG Style Transfer: ALL Domain1 → Domain4")
print("="*70)

# 获取源域和目标域的图像
source_domain_path = os.path.join(BASE_PATH, SOURCE_DOMAIN, "train", "imgs")
target_domain_path = os.path.join(BASE_PATH, TARGET_DOMAIN, "train", "imgs")

if not os.path.exists(source_domain_path):
    print(f"Error: Source domain path not found: {source_domain_path}")
    exit()

if not os.path.exists(target_domain_path):
    print(f"Error: Target domain path not found: {target_domain_path}")
    exit()

# 1. 选择代表性图像对来确定最佳参数
print("\nStep 1: Determining optimal parameters using representative samples...")
source_images = sorted([f for f in os.listdir(source_domain_path) if f.lower().endswith('.png')])
target_images = sorted([f for f in os.listdir(target_domain_path) if f.lower().endswith('.png')])

# 选择几个样本来确定最佳参数
num_param_samples = min(3, len(source_images), len(target_images))
param_source_samples = source_images[:num_param_samples]
param_target_samples = target_images[:num_param_samples]

best_scores = []
for i in range(num_param_samples):
    source_img_path = os.path.join(source_domain_path, param_source_samples[i])
    target_img_path = os.path.join(target_domain_path, param_target_samples[i])
    
    print(f"\n  Testing parameters with sample {i+1}:")
    print(f"    Source: {param_source_samples[i]}")
    print(f"    Target: {param_target_samples[i]}")
    
    best_params = find_best_parameters(source_img_path, target_img_path)
    best_scores.append(best_params)

# 使用平均参数
avg_lambda = np.mean([s['lambda'] for s in best_scores])
avg_mask_ratio = np.mean([s['mask_ratio'] for s in best_scores])

best_params = {
    'lambda': float(avg_lambda),
    'mask_ratio': float(avg_mask_ratio)
}

print(f"\nFinal parameters for all images:")
print(f"  Lambda (λ): {best_params['lambda']:.2f}")
print(f"  Mask ratio: {best_params['mask_ratio']:.2f}")

# 2. 处理所有图像
all_results, all_names = process_all_images(source_domain_path, target_domain_path, best_params)

if not all_results:
    print("No images were successfully processed")
    exit()

# 3. 选择代表性样本进行可视化
selected_results, selected_names = select_representative_samples(
    all_results, all_names, num_samples=NUM_VISUALIZATION_SAMPLES
)

# 4. 为每个选定样本创建详细可视化
print(f"\nCreating detailed visualizations for {len(selected_results)} selected samples...")
for i, (result, name) in enumerate(zip(selected_results, selected_names)):
    save_path = os.path.join(OUTPUT_DIR, f"detailed_sample_{i+1}.png")
    visualize_feddg_results(result, f"Sample {i+1}: {name}", save_path)
    print(f"  Detailed visualization saved: detailed_sample_{i+1}.png")

# 5. 创建对比可视化
create_comparison_visualization(selected_results, selected_names)

# 6. 保存参数配置
config_file = os.path.join(OUTPUT_DIR, "feddg_config_all.txt")
with open(config_file, 'w') as f:
    f.write("FedDG Configuration - All Images\n")
    f.write("="*50 + "\n\n")
    f.write(f"Source Domain: {SOURCE_DOMAIN}\n")
    f.write(f"Target Domain: {TARGET_DOMAIN}\n")
    f.write(f"Total images processed: {len(all_results)}\n")
    f.write(f"Images saved to: {MIGRATED_DIR}\n\n")
    
    f.write("Optimal Parameters:\n")
    f.write(f"  Lambda: {best_params['lambda']:.2f}\n")
    f.write(f"  Mask ratio: {best_params['mask_ratio']:.2f}\n\n")
    
    f.write("Selected Samples for Visualization:\n")
    for i, name in enumerate(selected_names):
        f.write(f"  {i+1}. {name}\n")
    
    f.write("\nOutput Files:\n")
    f.write(f"  Migrated images: migrated_*.png in {MIGRATED_DIR}\n")
    f.write(f"  Detailed visualizations: detailed_sample_*.png\n")
    f.write(f"  Comparison overview: comparison_overview.png\n")
    f.write(f"  Performance statistics: performance_statistics.png\n")

print(f"\nConfiguration saved to: {config_file}")

# ========== 10. 最终输出说明 ==========
print("\n" + "="*70)
print("FEDDG TRANSFER COMPLETE - ALL IMAGES")
print("="*70)

print(f"\nSummary:")
print(f"  • Processed {len(all_results)} images from {SOURCE_DOMAIN}")
print(f"  • Migrated to {TARGET_DOMAIN} style using FedDG")
print(f"  • All migrated images saved to: {os.path.abspath(MIGRATED_DIR)}")
print(f"  • Selected {len(selected_results)} representative samples for visualization")
print(f"  • Used parameters: lambda={best_params['lambda']:.2f}, mask_ratio={best_params['mask_ratio']:.2f}")

print(f"\nOutput directory: {os.path.abspath(OUTPUT_DIR)}")
print("\nGenerated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png') or f.endswith('.txt'):
        file_path = os.path.join(OUTPUT_DIR, f)
        file_size = os.path.getsize(file_path) / 1024
        print(f"  • {f} ({file_size:.1f} KB)")

print(f"\nMigrated images directory ({MIGRATED_DIR}):")
migrated_files = sorted(os.listdir(MIGRATED_DIR))
print(f"  • Contains {len(migrated_files)} migrated images")
if len(migrated_files) > 0:
    print(f"  • First 5 files: {', '.join(migrated_files[:5])}")
    if len(migrated_files) > 5:
        print(f"  • ... and {len(migrated_files)-5} more")

print("\n" + "="*70)
print("NEXT STEPS FOR TASK3:")
print("="*70)
print("""
1. Use migrated images in Task3:
   - Train segmentation model with Domain1 labels
   - Use migrated images as additional training data
   
2. Expected benefits:
   - Increased dataset diversity
   - Better generalization to Domain4
   - Improved segmentation performance on all domains
   
3. File organization for Task3:
   - Original Domain1 images: ../data/ODOC/Domain1/train/imgs/
   - Migrated images: feddg_migration_all/migrated_images/
   - Domain1 labels: ../data/ODOC/Domain1/train/masks/
   
4. Training strategy:
   - Combine original and migrated images
   - Use same labels for both (Domain1 labels)
   - Evaluate on all domains (Domain1-5 test sets)
""")
