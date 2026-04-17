import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 参数设置 ==========
BASE_PATH = "../data/ODOC"
DOMAINS = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5']
SAMPLES_PER_DOMAIN = 2  # 每个域展示的样本数量（减少以看得更清楚）
OUTPUT_DIR = "fft_visualization"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 2. 傅里叶变换辅助函数 ==========
def apply_2d_fft(image_path):
    """
    对图像进行2D傅里叶变换，返回幅度谱和相位谱
    包括：灰度转换、频谱中心化
    """
    try:
        # 1. 读取图像
        img = Image.open(image_path)
        
        # 2. 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        img_array = np.array(img, dtype=np.float32)
        
        # 3. 进行2D傅里叶变换
        f = np.fft.fft2(img_array)
        
        # 4. 频谱中心化（将零频移到中心）
        fshift = np.fft.fftshift(f)
        
        # 5. 计算幅度谱（取绝对值）
        magnitude = np.abs(fshift)
        
        # 6. 计算相位谱（取角度）
        phase = np.angle(fshift)
        
        return {
            'original': img_array,
            'magnitude': magnitude,  # 幅度谱（线性）
            'magnitude_log': np.log1p(magnitude),  # 幅度谱（对数变换）
            'phase': phase,  # 相位谱
            'fshift': fshift,
            'image_name': os.path.basename(image_path),
            'image_shape': img_array.shape
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# ========== 3. 专门展示幅度谱和相位谱的函数 ==========
def visualize_amplitude_phase_only(fft_results, domain_name, save_path):
    """
    专门展示幅度谱和相位谱，更清晰地显示频域信息
    """
    num_samples = len(fft_results)
    
    # 创建两行布局：第一行幅度谱，第二行相位谱
    fig, axes = plt.subplots(2, num_samples * 2, figsize=(6 * num_samples, 10))
    
    if num_samples == 1:
        axes = axes.reshape(2, 2)
    
    for i, result in enumerate(fft_results):
        if result is None:
            continue
        
        # 第一行：幅度谱
        # 左：线性幅度谱
        ax_mag_linear = axes[0, i*2]
        im_mag = ax_mag_linear.imshow(result['magnitude'], cmap='hot', aspect='auto')
        ax_mag_linear.set_title(f"Magnitude Spectrum (Linear)\n{result['image_name']}", fontsize=12)
        ax_mag_linear.axis('off')
        plt.colorbar(im_mag, ax=ax_mag_linear, fraction=0.046, pad=0.04)
        
        # 右：对数幅度谱
        ax_mag_log = axes[0, i*2+1]
        im_mag_log = ax_mag_log.imshow(result['magnitude_log'], cmap='hot', aspect='auto')
        ax_mag_log.set_title(f"Magnitude Spectrum (Log Scale)\n{result['image_name']}", fontsize=12)
        ax_mag_log.axis('off')
        plt.colorbar(im_mag_log, ax=ax_mag_log, fraction=0.046, pad=0.04)
        
        # 第二行：相位谱
        # 左：相位谱
        ax_phase = axes[1, i*2]
        im_phase = ax_phase.imshow(result['phase'], cmap='hsv', aspect='auto', 
                                  vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(f"Phase Spectrum\n{result['image_name']}", fontsize=12)
        ax_phase.axis('off')
        plt.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
        
        # 右：相位谱直方图
        ax_phase_hist = axes[1, i*2+1]
        phase_flat = result['phase'].flatten()
        ax_phase_hist.hist(phase_flat, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax_phase_hist.set_title(f"Phase Distribution\n{result['image_name']}", fontsize=12)
        ax_phase_hist.set_xlabel("Phase (radians)")
        ax_phase_hist.set_ylabel("Frequency")
        ax_phase_hist.grid(True, alpha=0.3)
    
    plt.suptitle(f"Frequency Domain Analysis - {domain_name}\n(Amplitude and Phase Spectra)", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Amplitude/Phase visualization saved to: {save_path}")

# ========== 4. 完整可视化函数 ==========
def visualize_complete_analysis(fft_results, domain_name, save_path):
    """
    完整的可视化：包括原始图像和频谱
    """
    num_samples = len(fft_results)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(fft_results):
        if result is None:
            continue
        
        # 第1列：原始图像
        axes[i, 0].imshow(result['original'], cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f"Original Image\n{result['image_name']}")
        axes[i, 0].axis('off')
        
        # 第2列：线性幅度谱
        im_mag = axes[i, 1].imshow(result['magnitude'], cmap='hot', aspect='auto')
        axes[i, 1].set_title("Magnitude Spectrum (Centered)")
        axes[i, 1].axis('off')
        plt.colorbar(im_mag, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # 第3列：对数幅度谱
        im_mag_log = axes[i, 2].imshow(result['magnitude_log'], cmap='hot', aspect='auto')
        axes[i, 2].set_title("Magnitude Spectrum (Log Scale)")
        axes[i, 2].axis('off')
        plt.colorbar(im_mag_log, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # 第4列：相位谱
        im_phase = axes[i, 3].imshow(result['phase'], cmap='hsv', aspect='auto', 
                                    vmin=-np.pi, vmax=np.pi)
        axes[i, 3].set_title("Phase Spectrum")
        axes[i, 3].axis('off')
        plt.colorbar(im_phase, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.suptitle(f"2D Fourier Transform Analysis - {domain_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path.replace('_amp_phase', '_complete'), dpi=150, bbox_inches='tight')
    plt.close()

# ========== 5. 频谱特性分析 ==========
def analyze_spectrum_features(fft_result, domain_name, image_name):
    """分析单个图像的频谱特征"""
    magnitude = fft_result['magnitude']
    phase = fft_result['phase']
    
    # 获取中心点（零频）
    center_y, center_x = magnitude.shape[0]//2, magnitude.shape[1]//2
    
    # 中心区域的能量（低频）
    center_radius = 10
    center_mask = np.zeros_like(magnitude, dtype=bool)
    for y in range(max(0, center_y-center_radius), min(magnitude.shape[0], center_y+center_radius)):
        for x in range(max(0, center_x-center_radius), min(magnitude.shape[1], center_x+center_radius)):
            if np.sqrt((y-center_y)**2 + (x-center_x)**2) <= center_radius:
                center_mask[y, x] = True
    
    center_energy = np.sum(magnitude[center_mask])
    total_energy = np.sum(magnitude)
    low_freq_ratio = center_energy / total_energy if total_energy > 0 else 0
    
    print(f"    Image: {image_name}")
    print(f"      Shape: {fft_result['image_shape']}")
    print(f"      Total energy: {total_energy:.2e}")
    print(f"      Low frequency energy ratio: {low_freq_ratio:.3%}")
    print(f"      Magnitude range: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
    print(f"      Phase range: [{phase.min():.3f}, {phase.max():.3f}] radians")

# ========== 6. 主处理流程 ==========
print("2D Fourier Transform Analysis - Amplitude and Phase Spectra")
print("="*70)

for domain_idx, domain in enumerate(DOMAINS):
    domain_path = os.path.join(BASE_PATH, domain, "train", "imgs")
    
    if not os.path.exists(domain_path):
        print(f"\nWarning: {domain_path} does not exist, skipping {domain}...")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {domain}")
    print(f"Path: {domain_path}")
    print('='*60)
    
    # 获取该域的所有PNG图像
    image_files = []
    for f in os.listdir(domain_path):
        if f.lower().endswith('.png'):
            image_files.append(os.path.join(domain_path, f))
    
    if not image_files:
        print(f"No PNG images found in {domain_path}")
        continue
    
    print(f"Found {len(image_files)} images")
    
    # 选择前几个样本
    selected_images = image_files[:SAMPLES_PER_DOMAIN]
    
    # 对每个选中的图像进行傅里叶变换
    fft_results = []
    for img_path in selected_images:
        print(f"\n  Processing: {os.path.basename(img_path)}")
        result = apply_2d_fft(img_path)
        if result is not None:
            fft_results.append(result)
            # 分析频谱特征
            analyze_spectrum_features(result, domain, os.path.basename(img_path))
    
    if not fft_results:
        print(f"No valid results for {domain}")
        continue
    
    # 1. 专门展示幅度谱和相位谱
    amp_phase_path = os.path.join(OUTPUT_DIR, f"amplitude_phase_{domain}.png")
    visualize_amplitude_phase_only(fft_results, domain, amp_phase_path)
    
    # 2. 完整分析可视化
    complete_path = os.path.join(OUTPUT_DIR, f"complete_analysis_{domain}.png")
    visualize_complete_analysis(fft_results, domain, complete_path)
    
    # 3. 单独保存幅度谱和相位谱图像
    for i, result in enumerate(fft_results):
        # 保存幅度谱
        plt.figure(figsize=(8, 6))
        plt.imshow(result['magnitude_log'], cmap='hot')
        plt.title(f"Amplitude Spectrum (Log) - {domain} - {result['image_name']}")
        plt.colorbar(label='Log Amplitude')
        plt.axis('off')
        mag_save_path = os.path.join(OUTPUT_DIR, f"amplitude_{domain}_{result['image_name'].replace('.png', '')}.png")
        plt.savefig(mag_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存相位谱
        plt.figure(figsize=(8, 6))
        plt.imshow(result['phase'], cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.title(f"Phase Spectrum - {domain} - {result['image_name']}")
        plt.colorbar(label='Phase (radians)')
        plt.axis('off')
        phase_save_path = os.path.join(OUTPUT_DIR, f"phase_{domain}_{result['image_name'].replace('.png', '')}.png")
        plt.savefig(phase_save_path, dpi=150, bbox_inches='tight')
        plt.close()

# ========== 7. 频谱对比分析 ==========
print("\n" + "="*70)
print("SPECTRUM COMPARISON ACROSS DOMAINS")
print("="*70)

# 收集所有域的频谱数据进行比较
all_domain_fft = {}
for domain in DOMAINS:
    amp_phase_file = os.path.join(OUTPUT_DIR, f"amplitude_phase_{domain}.png")
    if os.path.exists(amp_phase_file):
        all_domain_fft[domain] = amp_phase_file

if len(all_domain_fft) >= 2:
    print("\nGenerated amplitude and phase visualizations:")
    for domain, file_path in all_domain_fft.items():
        print(f"  {domain}: {file_path}")
    
    # 创建汇总图
    fig, axes = plt.subplots(len(all_domain_fft), 1, figsize=(12, 4*len(all_domain_fft)))
    
    if len(all_domain_fft) == 1:
        axes = [axes]
    
    for idx, (domain, file_path) in enumerate(all_domain_fft.items()):
        # 加载并显示第一个样本
        img = plt.imread(file_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"{domain} - Amplitude and Phase Spectra", fontsize=14)
    
    plt.suptitle("Comparison of Amplitude and Phase Spectra Across Domains", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "spectra_comparison_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary comparison saved to: {summary_path}")

# ========== 8. 最终说明 ==========
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

output_files = []
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.png'):
        output_files.append(f)

print(f"\nGenerated {len(output_files)} visualization files in: {os.path.abspath(OUTPUT_DIR)}")

print("\nKey output files (amplitude and phase spectra):")
for domain in DOMAINS:
    amp_phase_file = f"amplitude_phase_{domain}.png"
    if amp_phase_file in output_files:
        print(f"  • {amp_phase_file}")
        print(f"    - Shows both amplitude and phase spectra for {domain}")
        print(f"    - Top row: Amplitude spectrum (linear and log scale)")
        print(f"    - Bottom row: Phase spectrum and distribution")

print("""
SPECTRUM INTERPRETATION GUIDE:
--------------------------------
1. AMPLITUDE SPECTRUM (Magnitude Spectrum):
   - Shows the strength of each frequency component
   - Center (low frequencies): Overall image brightness and structure
   - Edges (high frequencies): Fine details, edges, and textures
   - Log scale enhances visualization of weaker components

2. PHASE SPECTRUM:
   - Shows the phase angle of each frequency component
   - Contains structural information about the image
   - Phase is critical for image reconstruction
   - Displayed in HSV colormap (-π to π radians)

3. SPECTRUM CENTERING:
   - Zero frequency (DC component) is shifted to the center
   - This is done using np.fft.fftshift()
   - Makes low frequencies visible in the center

4. GRAYSCALE ADJUSTMENT:
   - All images are converted to grayscale before FFT
   - This simplifies analysis to a single channel
""")

# 保存分析说明
with open(os.path.join(OUTPUT_DIR, "README.txt"), "w") as f:
    f.write("2D FOURIER TRANSFORM ANALYSIS - AMPLITUDE AND PHASE SPECTRA\n")
    f.write("="*60 + "\n\n")
    f.write("This directory contains visualizations of 2D Fourier transforms.\n\n")
    f.write("KEY FILES:\n")
    f.write("1. amplitude_phase_<domain>.png - Main visualization of amplitude and phase\n")
    f.write("2. complete_analysis_<domain>.png - Complete analysis including original image\n")
    f.write("3. amplitude_<domain>_<image>.png - Individual amplitude spectrum\n")
    f.write("4. phase_<domain>_<image>.png - Individual phase spectrum\n")
    f.write("5. spectra_comparison_summary.png - Cross-domain comparison\n\n")
    f.write("INTERPRETATION:\n")
    f.write("- Amplitude spectrum shows frequency strength\n")
    f.write("- Phase spectrum shows frequency phase angles\n")
    f.write("- Center of images = low frequencies\n")
    f.write("- Edges of images = high frequencies\n")

print(f"\nDetailed guide saved to: {os.path.join(OUTPUT_DIR, 'README.txt')}")