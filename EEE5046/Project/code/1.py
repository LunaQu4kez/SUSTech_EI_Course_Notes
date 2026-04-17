import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 参数设置 ==========
BASE_PATH = "../data/ODOC"
DOMAINS = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5']
IMAGE_SIZE = 224
BATCH_SIZE = 32
PERPLEXITY = 30  # t-SNE 参数，适合中等规模数据
RANDOM_STATE = 42

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ========== 2. 加载预训练模型作为特征提取器 ==========
print("Loading pre-trained ResNet18...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)
model.eval()

# 去掉最后一层分类器，保留特征提取部分
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# ========== 3. 图像预处理 ==========
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ========== 4. 提取图像特征 ==========
def extract_features(image_paths, batch_size=BATCH_SIZE):
    """批量提取图像特征"""
    features = []
    num_images = len(image_paths)
    
    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0)  # [1, C, H, W]
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
            
        batch_tensor = torch.cat(batch_images, dim=0).to(device)
        
        with torch.no_grad():
            batch_features = feature_extractor(batch_tensor)
            batch_features = batch_features.squeeze().cpu().numpy()
            
            # 处理 batch_size=1 的情况
            if batch_features.ndim == 1:
                batch_features = batch_features.reshape(1, -1)
                
            features.append(batch_features)
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed {min(i+batch_size, num_images)}/{num_images} images...")
    
    return np.vstack(features) if features else np.array([])

# ========== 5. 收集所有域的图像路径 ==========
print("Collecting image paths...")
all_image_paths = []
all_labels = []
domain_colors = {}
color_palette = ['red', 'blue', 'green', 'orange', 'purple']

for idx, domain in enumerate(DOMAINS):
    domain_path = os.path.join(BASE_PATH, domain, "train", "imgs")
    
    if not os.path.exists(domain_path):
        print(f"Warning: {domain_path} does not exist, skipping...")
        continue
    
    images = []
    for f in os.listdir(domain_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(os.path.join(domain_path, f))
    
    if not images:
        print(f"Warning: No images found in {domain_path}")
        continue
    
    print(f"Domain {domain}: {len(images)} images found")
    
    all_image_paths.extend(images)
    all_labels.extend([idx] * len(images))
    domain_colors[idx] = (domain, color_palette[idx])

# ========== 6. 检查是否有图像 ==========
if len(all_image_paths) == 0:
    print("No images found! Please check your data path.")
    exit()

print(f"\nTotal images: {len(all_image_paths)}")
print("Starting feature extraction...")

# ========== 7. 提取特征 ==========
features = extract_features(all_image_paths)

if len(features) == 0:
    print("No features extracted!")
    exit()

print(f"Features shape: {features.shape}")

# ========== 8. 特征标准化 ==========
print("Standardizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ========== 9. t-SNE 降维 ==========
print("Running t-SNE...")

# 检查 scikit-learn 版本，使用正确的参数名
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

# 新版本的 scikit-learn 使用 max_iter，旧版本使用 n_iter
try:
    # 尝试使用新版本参数
    tsne = TSNE(
        n_components=2,
        perplexity=min(PERPLEXITY, len(features_scaled)-1),
        random_state=RANDOM_STATE,
        max_iter=1000,  # 新版本使用 max_iter
        learning_rate=200,
        verbose=1,
        init='pca'  # 使用 PCA 初始化，结果更稳定
    )
    print("Using max_iter parameter (newer scikit-learn version)")
except TypeError:
    # 如果失败，使用旧版本参数
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=min(PERPLEXITY, len(features_scaled)-1),
            random_state=RANDOM_STATE,
            n_iter=1000,  # 旧版本使用 n_iter
            learning_rate=200,
            verbose=1,
            init='pca'
        )
        print("Using n_iter parameter (older scikit-learn version)")
    except Exception as e:
        # 使用最简单的参数
        print(f"Error with both parameter names: {e}")
        print("Using basic parameters...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(PERPLEXITY, len(features_scaled)-1),
            random_state=RANDOM_STATE,
            verbose=1
        )

features_tsne = tsne.fit_transform(features_scaled)

# ========== 10. 可视化 ==========
print("Creating visualization...")
plt.figure(figsize=(12, 10))

# 为每个域绘制散点图
for label in set(all_labels):
    idx = [i for i, l in enumerate(all_labels) if l == label]
    domain_name, color = domain_colors[label]
    
    plt.scatter(
        features_tsne[idx, 0], 
        features_tsne[idx, 1], 
        c=color, 
        label=domain_name, 
        alpha=0.6,
        s=30,
        edgecolors='w',
        linewidth=0.5
    )

plt.legend(fontsize=12, markerscale=2)
plt.title("t-SNE Visualization of ODOC Domains 1-5 (Test Images)", fontsize=16, fontweight='bold')
plt.xlabel("t-SNE Dimension 1", fontsize=14)
plt.ylabel("t-SNE Dimension 2", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
output_path = "tsne_domains_odoc.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved as: {output_path}")

# 显示图片
plt.show()

# ========== 11. 打印统计信息 ==========
print("\n" + "="*50)
print("Visualization Summary:")
print("="*50)
for label in sorted(set(all_labels)):
    domain_name, _ = domain_colors[label]
    count = sum(1 for l in all_labels if l == label)
    print(f"{domain_name}: {count} images")

print(f"\nTotal domains: {len(set(all_labels))}")
print(f"Total images: {len(all_image_paths)}")
print(f"Feature dimension: {features.shape[1]}")
print(f"t-SNE shape: {features_tsne.shape}")
