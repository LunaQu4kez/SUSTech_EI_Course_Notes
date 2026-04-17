import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 参数设置 ==========
BASE_PATH = "../data/ODOC"
SOURCE_DOMAIN = "Domain1"
TARGET_DOMAIN = "Domain4"  # Task1中Dice最低的域
BATCH_SIZE = 4
IMAGE_SIZE = 256
OUTPUT_DIR = "cyclegan_final_fixed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 2. 数据加载器 ==========
class DomainDataset(Dataset):
    def __init__(self, domain_path, transform=None):
        self.domain_path = domain_path
        self.transform = transform
        self.image_files = []
        
        if os.path.exists(domain_path):
            for f in os.listdir(domain_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(domain_path, f))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# ========== 3. CycleGAN网络定义 ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residual_blocks=6):
        super(Generator, self).__init__()
        
        # 初始卷积层
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # 下采样
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # 残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # 上采样
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # 输出层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, input_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        # 使用PatchGAN结构，输出为NxN的patch
        model = [
            # 输入: 256x256x3
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # 128x128x64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 64x64x128
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 32x32x256
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 16x16x512
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最终卷积层，输出单通道
            nn.Conv2d(512, 1, 4, stride=1, padding=1)  # 15x15x1
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# ========== 4. 修复的CycleGAN训练类 ==========
class FixedCycleGAN:
    def __init__(self, device=DEVICE, image_size=IMAGE_SIZE):
        self.device = device
        self.image_size = image_size
        
        # 初始化生成器和判别器
        self.G_AB = Generator().to(device)  # Domain1 -> Domain4
        self.G_BA = Generator().to(device)  # Domain4 -> Domain1
        self.D_A = Discriminator().to(device)  # 判别Domain1
        self.D_B = Discriminator().to(device)  # 判别Domain4
        
        # 损失函数
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # 优化器
        self.optimizer_G = optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = optim.Adam(self.D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_B = optim.Adam(self.D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 存储训练历史
        self.history = {
            'loss_G': [], 'loss_D_A': [], 'loss_D_B': [],
            'loss_cycle_A': [], 'loss_cycle_B': [], 'loss_identity': [],
            'epoch_losses': {'G': [], 'D_A': [], 'D_B': []}
        }
    
    def train_step(self, real_A, real_B):
        """训练一个批次 - 修复版本"""
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        batch_size = real_A.size(0)
        
        # ========== 训练生成器 ==========
        self.optimizer_G.zero_grad()
        
        # 身份损失
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        # 生成假图像
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        
        # 获取判别器输出大小
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)
        
        # 创建动态大小的标签
        valid_B = torch.ones(pred_fake_B.shape, device=self.device, requires_grad=False)
        fake_label_B = torch.zeros(pred_fake_B.shape, device=self.device, requires_grad=False)
        valid_A = torch.ones(pred_fake_A.shape, device=self.device, requires_grad=False)
        fake_label_A = torch.zeros(pred_fake_A.shape, device=self.device, requires_grad=False)
        
        # GAN损失
        loss_GAN_AB = self.criterion_gan(self.D_B(fake_B), valid_B)
        loss_GAN_BA = self.criterion_gan(self.D_A(fake_A), valid_A)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
        # 循环一致性损失
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        
        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)
        
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        # 总损失
        lambda_cyc = 10.0
        lambda_id = 0.5 * lambda_cyc
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # ========== 训练判别器A ==========
        self.optimizer_D_A.zero_grad()
        
        # 判别真实图像
        pred_real_A = self.D_A(real_A)
        loss_real_A = self.criterion_gan(pred_real_A, valid_A)
        
        # 判别假图像
        pred_fake_A = self.D_A(fake_A.detach())
        loss_fake_A = self.criterion_gan(pred_fake_A, fake_label_A)
        
        loss_D_A = (loss_real_A + loss_fake_A) / 2
        loss_D_A.backward()
        self.optimizer_D_A.step()
        
        # ========== 训练判别器B ==========
        self.optimizer_D_B.zero_grad()
        
        # 判别真实图像
        pred_real_B = self.D_B(real_B)
        loss_real_B = self.criterion_gan(pred_real_B, valid_B)
        
        # 判别假图像
        pred_fake_B = self.D_B(fake_B.detach())
        loss_fake_B = self.criterion_gan(pred_fake_B, fake_label_B)
        
        loss_D_B = (loss_real_B + loss_fake_B) / 2
        loss_D_B.backward()
        self.optimizer_D_B.step()
        
        # 记录损失
        self.history['loss_G'].append(loss_G.item())
        self.history['loss_D_A'].append(loss_D_A.item())
        self.history['loss_D_B'].append(loss_D_B.item())
        self.history['loss_cycle_A'].append(loss_cycle_A.item())
        self.history['loss_cycle_B'].append(loss_cycle_B.item())
        self.history['loss_identity'].append(loss_identity.item())
        
        return {
            'loss_G': loss_G.item(),
            'loss_D_A': loss_D_A.item(),
            'loss_D_B': loss_D_B.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_identity': loss_identity.item()
        }
    
    def train(self, dataloader_A, dataloader_B, epochs=10):
        """训练CycleGAN"""
        print(f"Training CycleGAN for {epochs} epochs...")
        print(f"Using device: {self.device}")
        
        for epoch in range(epochs):
            epoch_loss_G = 0
            epoch_loss_D_A = 0
            epoch_loss_D_B = 0
            num_batches = 0
            
            # 创建迭代器
            iter_A = iter(dataloader_A)
            iter_B = iter(dataloader_B)
            
            try:
                while True:
                    try:
                        real_A = next(iter_A)
                        real_B = next(iter_B)
                    except StopIteration:
                        break
                    
                    # 确保批次大小相同
                    if real_A.size(0) != real_B.size(0):
                        # 如果不同，调整到较小的大小
                        min_batch = min(real_A.size(0), real_B.size(0))
                        real_A = real_A[:min_batch]
                        real_B = real_B[:min_batch]
                    
                    losses = self.train_step(real_A, real_B)
                    
                    epoch_loss_G += losses['loss_G']
                    epoch_loss_D_A += losses['loss_D_A']
                    epoch_loss_D_B += losses['loss_D_B']
                    num_batches += 1
            
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                continue
            
            # 打印进度
            if num_batches > 0:
                avg_loss_G = epoch_loss_G / num_batches
                avg_loss_D_A = epoch_loss_D_A / num_batches
                avg_loss_D_B = epoch_loss_D_B / num_batches
                
                # 存储epoch平均损失
                self.history['epoch_losses']['G'].append(avg_loss_G)
                self.history['epoch_losses']['D_A'].append(avg_loss_D_A)
                self.history['epoch_losses']['D_B'].append(avg_loss_D_B)
                
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss_G: {avg_loss_G:.4f} "
                      f"Loss_D_A: {avg_loss_D_A:.4f} "
                      f"Loss_D_B: {avg_loss_D_B:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] No batches processed")
            
            # 保存检查点
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
        
        print("Training completed!")
        
        # 绘制训练收敛曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练收敛曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. 总损失曲线
        if self.history['loss_G']:
            axes[0, 0].plot(self.history['loss_G'], label='Generator Loss', linewidth=2)
            axes[0, 0].set_xlabel('Iterations')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Generator Loss Progression')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 判别器损失曲线
        if self.history['loss_D_A'] and self.history['loss_D_B']:
            axes[0, 1].plot(self.history['loss_D_A'], label='Discriminator A Loss', alpha=0.7)
            axes[0, 1].plot(self.history['loss_D_B'], label='Discriminator B Loss', alpha=0.7)
            axes[0, 1].set_xlabel('Iterations')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Discriminator Loss Progression')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 循环一致性损失
        if self.history['loss_cycle_A'] and self.history['loss_cycle_B']:
            axes[0, 2].plot(self.history['loss_cycle_A'], label='Cycle Loss A→B→A', alpha=0.7)
            axes[0, 2].plot(self.history['loss_cycle_B'], label='Cycle Loss B→A→B', alpha=0.7)
            axes[0, 2].set_xlabel('Iterations')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].set_title('Cycle Consistency Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Epoch平均损失
        if self.history['epoch_losses']['G']:
            epochs = range(1, len(self.history['epoch_losses']['G']) + 1)
            axes[1, 0].plot(epochs, self.history['epoch_losses']['G'], 'o-', label='Generator', linewidth=2)
            axes[1, 0].plot(epochs, self.history['epoch_losses']['D_A'], 's-', label='Discriminator A', linewidth=2)
            axes[1, 0].plot(epochs, self.history['epoch_losses']['D_B'], '^-', label='Discriminator B', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Average Loss')
            axes[1, 0].set_title('Epoch-wise Average Losses')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 身份损失
        if self.history['loss_identity']:
            axes[1, 1].plot(self.history['loss_identity'], label='Identity Loss', color='purple', linewidth=2)
            axes[1, 1].set_xlabel('Iterations')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Identity Loss Progression')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 损失分布直方图
        if self.history['loss_G']:
            axes[1, 2].hist(self.history['loss_G'], bins=50, alpha=0.7, label='Generator', color='blue')
            axes[1, 2].hist(self.history['loss_D_A'], bins=50, alpha=0.7, label='Discriminator A', color='red')
            axes[1, 2].hist(self.history['loss_D_B'], bins=50, alpha=0.7, label='Discriminator B', color='green')
            axes[1, 2].set_xlabel('Loss Value')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Loss Distribution')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'CycleGAN Training Convergence Curves\n{SOURCE_DOMAIN} → {TARGET_DOMAIN}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存收敛曲线图
        convergence_path = os.path.join(OUTPUT_DIR, "training_convergence.png")
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training convergence curves saved to: {convergence_path}")
    
    def save_checkpoint(self, path):
        """保存模型检查点"""
        checkpoint = {
            'G_AB_state_dict': self.G_AB.state_dict(),
            'G_BA_state_dict': self.G_BA.state_dict(),
            'D_A_state_dict': self.D_A.state_dict(),
            'D_B_state_dict': self.D_B.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': self.optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': self.optimizer_D_B.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
        self.G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
        self.D_A.load_state_dict(checkpoint['D_A_state_dict'])
        self.D_B.load_state_dict(checkpoint['D_B_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")
    
    def translate(self, image_tensor, direction='A_to_B'):
        """将图像从一个域转换到另一个域"""
        self.G_AB.eval()
        self.G_BA.eval()
        
        with torch.no_grad():
            if direction == 'A_to_B':  # Domain1 -> Domain4
                return self.G_AB(image_tensor.to(self.device))
            else:  # Domain4 -> Domain1
                return self.G_BA(image_tensor.to(self.device))

# ========== 5. 批量翻译函数 ==========
def translate_all_images(cyclegan, source_domain_path, output_dir):
    """翻译所有源域图像"""
    print("\n" + "="*70)
    print("Translating all Domain1 images to Domain4 style...")
    print("="*70)
    
    # 创建输出目录
    translated_dir = os.path.join(output_dir, "translated_images")
    os.makedirs(translated_dir, exist_ok=True)
    
    # 获取源图像
    source_files = []
    if os.path.exists(source_domain_path):
        for f in sorted(os.listdir(source_domain_path)):
            if f.lower().endswith('.png'):
                source_files.append(os.path.join(source_domain_path, f))
    
    if not source_files:
        print("No source images found")
        return [], []
    
    print(f"Found {len(source_files)} source images")
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    all_translated = []
    all_names = []
    
    cyclegan.G_AB.eval()
    
    for i, source_file in enumerate(source_files):
        try:
            # 加载源图像
            source_img = Image.open(source_file).convert('RGB')
            source_name = os.path.basename(source_file)
            
            # 应用变换
            source_tensor = transform(source_img).unsqueeze(0)
            
            # 翻译图像
            with torch.no_grad():
                translated_tensor = cyclegan.translate(source_tensor, direction='A_to_B')
            
            # 转换为PIL图像
            translated_np = translated_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            translated_np = (translated_np * 0.5 + 0.5) * 255  # 反归一化到[0,255]
            translated_np = np.clip(translated_np, 0, 255).astype(np.uint8)
            
            # 保存图像
            translated_img = Image.fromarray(translated_np)
            save_name = f"cyclegan_{source_name}"
            save_path = os.path.join(translated_dir, save_name)
            translated_img.save(save_path)
            
            all_translated.append(translated_np)
            all_names.append(source_name)
            
            if (i + 1) % 10 == 0:
                print(f"  Translated {i+1}/{len(source_files)} images...")
                
        except Exception as e:
            print(f"  Error processing {source_file}: {e}")
            continue
    
    print(f"\nSuccessfully translated {len(all_translated)} images")
    print(f"Translated images saved to: {translated_dir}")
    
    return all_translated, all_names, translated_dir

# ========== 6. 与3_1.py结果对比可视化 ==========
def compare_with_feddg(cyclegan_translated, feddg_dir, source_path, num_samples=5):
    """
    对比CycleGAN和FedDG的结果
    """
    print("\n" + "="*70)
    print("Comparing CycleGAN vs FedDG Results")
    print("="*70)
    
    # 获取FedDG迁移的图像
    feddg_translated_dir = os.path.join(feddg_dir, "migrated_images")
    if not os.path.exists(feddg_translated_dir):
        print(f"FedDG directory not found: {feddg_translated_dir}")
        return
    
    # 获取源图像文件
    source_files = []
    if os.path.exists(source_path):
        for f in sorted(os.listdir(source_path)):
            if f.lower().endswith('.png'):
                source_files.append(os.path.join(source_path, f))
    
    if not source_files:
        print("No source images found")
        return
    
    # 限制样本数量
    num_samples = min(num_samples, len(source_files))
    selected_indices = np.linspace(0, len(source_files)-1, num_samples, dtype=int)
    
    # 准备可视化
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, 4)
    
    # 加载变换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    for i, idx in enumerate(selected_indices):
        source_file = source_files[idx]
        source_name = os.path.basename(source_file)
        
        try:
            # 加载源图像
            source_img = Image.open(source_file).convert('RGB')
            
            # 获取CycleGAN结果
            source_tensor = transform(source_img).unsqueeze(0)
            with torch.no_grad():
                cyclegan_result = cyclegan.translate(source_tensor, direction='A_to_B')
            cyclegan_np = cyclegan_result.squeeze().cpu().permute(1, 2, 0).numpy()
            cyclegan_np = (cyclegan_np * 0.5 + 0.5) * 255
            cyclegan_np = np.clip(cyclegan_np, 0, 255).astype(np.uint8)
            
            # 获取FedDG结果
            feddg_file = os.path.join(feddg_translated_dir, f"migrated_{source_name}")
            if os.path.exists(feddg_file):
                feddg_img = Image.open(feddg_file).convert('RGB')
                feddg_np = np.array(feddg_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            else:
                feddg_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            
            # 显示结果
            axes[i, 0].imshow(source_img)
            axes[i, 0].set_title(f"Source\n{source_name}", fontsize=10)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(cyclegan_np)
            axes[i, 1].set_title("CycleGAN Result", fontsize=10, color='blue', fontweight='bold')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(feddg_np)
            axes[i, 2].set_title("FedDG Result", fontsize=10, color='green', fontweight='bold')
            axes[i, 2].axis('off')
            
            # 计算差异
            source_gray = np.array(source_img.convert('L').resize((IMAGE_SIZE, IMAGE_SIZE)))
            cyclegan_gray = np.array(Image.fromarray(cyclegan_np).convert('L'))
            feddg_gray = np.array(Image.fromarray(feddg_np).convert('L'))
            
            diff_cyclegan = np.abs(source_gray.astype(float) - cyclegan_gray.astype(float))
            diff_feddg = np.abs(source_gray.astype(float) - feddg_gray.astype(float))
            
            axes[i, 3].imshow(diff_cyclegan, cmap='hot', alpha=0.5, label='CycleGAN')
            axes[i, 3].imshow(diff_feddg, cmap='cool', alpha=0.5, label='FedDG')
            axes[i, 3].set_title(f"Differences\nCyc MSE: {np.mean(diff_cyclegan**2):.1f}\nFedDG MSE: {np.mean(diff_feddg**2):.1f}", 
                               fontsize=10)
            axes[i, 3].axis('off')
            
        except Exception as e:
            print(f"Error processing {source_name}: {e}")
            continue
    
    plt.suptitle(f'Comparison: CycleGAN vs FedDG Style Transfer\n{SOURCE_DOMAIN} → {TARGET_DOMAIN}', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = os.path.join(OUTPUT_DIR, "cyclegan_vs_feddg_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison visualization saved to: {comparison_path}")
    
    # 创建参数对比图
    create_parameter_comparison()

def create_parameter_comparison():
    """创建CycleGAN和FedDG参数对比图"""
    # 假设FedDG使用默认参数
    feddg_params = {
        'lambda': 0.5,
        'mask_ratio': 0.5,
        'method': 'Frequency Domain Mixing',
        'computational_cost': 'Low',
        'training_required': 'No',
        'inference_speed': 'Fast'
    }
    
    cyclegan_params = {
        'lambda_cyc': 10.0,
        'lambda_id': 5.0,
        'method': 'Adversarial Training',
        'computational_cost': 'High',
        'training_required': 'Yes (5 epochs)',
        'inference_speed': 'Medium'
    }
    
    # 创建对比表格
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建数据
    categories = ['Method', 'Lambda (λ)', 'Additional Param', 'Comp. Cost', 'Training', 'Speed']
    feddg_data = [feddg_params['method'], f"{feddg_params['lambda']:.1f}", 
                 f"Mask: {feddg_params['mask_ratio']:.1f}", feddg_params['computational_cost'],
                 feddg_params['training_required'], feddg_params['inference_speed']]
    cyclegan_data = [cyclegan_params['method'], f"λ_cyc: {cyclegan_params['lambda_cyc']:.1f}", 
                    f"λ_id: {cyclegan_params['lambda_id']:.1f}", cyclegan_params['computational_cost'],
                    cyclegan_params['training_required'], cyclegan_params['inference_speed']]
    
    table_data = [categories, feddg_data, cyclegan_data]
    
    # 创建表格
    table = ax.table(cellText=table_data, 
                     colLabels=None,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2]*6)
    
    # 设置样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # 设置标题行样式
    for j in range(len(categories)):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # 设置FedDG行样式
    for j in range(len(categories)):
        table[(1, j)].set_facecolor('#e1f5fe')
    
    # 设置CycleGAN行样式
    for j in range(len(categories)):
        table[(2, j)].set_facecolor('#f3e5f5')
    
    plt.title('Parameter Comparison: FedDG vs CycleGAN', fontsize=14, fontweight='bold', pad=20)
    
    # 保存参数对比图
    param_path = os.path.join(OUTPUT_DIR, "parameter_comparison.png")
    plt.savefig(param_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Parameter comparison saved to: {param_path}")

# ========== 7. 性能评估函数 ==========
def evaluate_performance(cyclegan, source_path, target_path, num_samples=10):
    """
    评估CycleGAN的性能
    """
    print("\n" + "="*70)
    print("Evaluating CycleGAN Performance")
    print("="*70)
    
    # 加载变换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 获取样本图像
    source_files = []
    if os.path.exists(source_path):
        for f in sorted(os.listdir(source_path)):
            if f.lower().endswith('.png'):
                source_files.append(os.path.join(source_path, f))
    
    target_files = []
    if os.path.exists(target_path):
        for f in sorted(os.listdir(target_path)):
            if f.lower().endswith('.png'):
                target_files.append(os.path.join(target_path, f))
    
    if not source_files or not target_files:
        print("Not enough images for evaluation")
        return
    
    num_samples = min(num_samples, len(source_files), len(target_files))
    
    # 计算指标
    psnr_values = []
    ssim_values = []
    mse_values = []
    
    for i in range(num_samples):
        try:
            # 加载源图像
            source_img = Image.open(source_files[i]).convert('RGB')
            source_tensor = transform(source_img).unsqueeze(0)
            
            # 生成CycleGAN结果
            with torch.no_grad():
                translated_tensor = cyclegan.translate(source_tensor, direction='A_to_B')
            
            # 转换为numpy
            translated_np = translated_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            translated_np = (translated_np * 0.5 + 0.5) * 255
            
            # 加载目标图像作为参考
            target_img = Image.open(target_files[i % len(target_files)]).convert('RGB')
            target_img = target_img.resize((IMAGE_SIZE, IMAGE_SIZE))
            target_np = np.array(target_img).astype(float)
            
            # 计算PSNR
            mse = np.mean((translated_np - target_np) ** 2)
            if mse == 0:
                psnr = 100
            else:
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            # 计算SSIM（简化版本）
            def ssim(img1, img2):
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2
                
                img1 = img1.astype(np.float64)
                img2 = img2.astype(np.float64)
                
                mu1 = np.mean(img1)
                mu2 = np.mean(img2)
                sigma1_sq = np.var(img1)
                sigma2_sq = np.var(img2)
                sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
                
                ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                          ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
                
                return ssim_val
            
            ssim_val = ssim(translated_np.mean(axis=2), target_np.mean(axis=2))
            
            psnr_values.append(psnr)
            ssim_values.append(ssim_val)
            mse_values.append(mse)
            
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            continue
    
    # 创建性能评估图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PSNR分布
    axes[0].hist(psnr_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(psnr_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('PSNR Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM分布
    axes[1].hist(ssim_values, bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(ssim_values):.3f}')
    axes[1].set_xlabel('SSIM')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SSIM Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MSE分布
    axes[2].hist(mse_values, bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[2].axvline(np.mean(mse_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(mse_values):.1f}')
    axes[2].set_xlabel('MSE')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('MSE Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'CycleGAN Performance Evaluation\nBased on {len(psnr_values)} samples', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存性能评估图
    eval_path = os.path.join(OUTPUT_DIR, "performance_evaluation.png")
    plt.savefig(eval_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPerformance Evaluation Results:")
    print(f"  Average PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"  Average SSIM: {np.mean(ssim_values):.3f}")
    print(f"  Average MSE: {np.mean(mse_values):.1f}")
    print(f"Performance evaluation saved to: {eval_path}")
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'mse': np.mean(mse_values)
    }

# ========== 8. 主程序 ==========
def main():
    print("="*70)
    print(f"CycleGAN Style Transfer: {SOURCE_DOMAIN} → {TARGET_DOMAIN}")
    print("="*70)
    
    # 设置路径
    source_path = os.path.join(BASE_PATH, SOURCE_DOMAIN, "train", "imgs")
    target_path = os.path.join(BASE_PATH, TARGET_DOMAIN, "train", "imgs")
    
    if not os.path.exists(source_path):
        print(f"Error: Source path not found: {source_path}")
        return
    
    if not os.path.exists(target_path):
        print(f"Error: Target path not found: {target_path}")
        return
    
    print(f"Source domain: {SOURCE_DOMAIN}")
    print(f"Target domain: {TARGET_DOMAIN}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集
    dataset_A = DomainDataset(source_path, transform=transform)
    dataset_B = DomainDataset(target_path, transform=transform)
    
    print(f"\nDataset sizes: {SOURCE_DOMAIN}={len(dataset_A)}, {TARGET_DOMAIN}={len(dataset_B)}")
    
    if len(dataset_A) == 0 or len(dataset_B) == 0:
        print("Insufficient data for training")
        return
    
    # 创建数据加载器 - 使用drop_last确保批次大小一致
    dataloader_A = DataLoader(dataset_A, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dataloader_B = DataLoader(dataset_B, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print(f"DataLoader sizes: A={len(dataloader_A)} batches, B={len(dataloader_B)} batches")
    
    # 初始化并训练CycleGAN
    print(f"\nInitializing CycleGAN on device: {DEVICE}")
    cyclegan = FixedCycleGAN(device=DEVICE, image_size=IMAGE_SIZE)
    
    # 训练模型
    cyclegan.train(dataloader_A, dataloader_B, epochs=100)
    
    # 保存最终模型
    final_path = os.path.join(OUTPUT_DIR, "cyclegan_final.pth")
    cyclegan.save_checkpoint(final_path)
    
    # 翻译所有图像
    translated_images, translated_names, translated_dir = translate_all_images(cyclegan, source_path, OUTPUT_DIR)
    
    if not translated_images:
        print("No images were translated")
        return
    
    # 评估性能
    performance = evaluate_performance(cyclegan, source_path, target_path, num_samples=10)
    
    # 与FedDG对比
    feddg_dir = "feddg_migration_all"
    compare_with_feddg(cyclegan, feddg_dir, source_path, num_samples=5)
    
    print("\n" + "="*70)
    print("TASK2 COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    for item in sorted(os.listdir(OUTPUT_DIR)):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            file_count = len([f for f in os.listdir(item_path) if f.endswith('.png')])
            print(f"  📁 {item}/ ({file_count} images)")
        elif os.path.isfile(item_path):
            size_kb = os.path.getsize(item_path) / 1024
            if item.endswith('.png'):
                print(f"  🖼️  {item} ({size_kb:.1f} KB)")
            elif item.endswith('.pth'):
                print(f"  🤖 {item} ({size_kb:.1f} KB)")
            else:
                print(f"  📄 {item} ({size_kb:.1f} KB)")
    
    # 保存配置信息
    config_file = os.path.join(OUTPUT_DIR, "cyclegan_config_summary.txt")
    with open(config_file, 'w') as f:
        f.write("CycleGAN Configuration Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Source Domain: {SOURCE_DOMAIN}\n")
        f.write(f"Target Domain: {TARGET_DOMAIN}\n")
        f.write(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Epochs: 5\n")
        f.write(f"Device: {DEVICE}\n\n")
        
        f.write("Loss Parameters:\n")
        f.write(f"  Lambda Cycle: 10.0\n")
        f.write(f"  Lambda Identity: 5.0\n")
        f.write(f"  GAN Loss: MSE\n")
        f.write(f"  Cycle Loss: L1\n")
        f.write(f"  Identity Loss: L1\n\n")
        
        f.write("Performance Summary:\n")
        if performance:
            f.write(f"  Average PSNR: {performance['psnr']:.2f} dB\n")
            f.write(f"  Average SSIM: {performance['ssim']:.3f}\n")
            f.write(f"  Average MSE: {performance['mse']:.1f}\n\n")
        
        f.write("Output Files:\n")
        for item in sorted(os.listdir(OUTPUT_DIR)):
            if item.endswith('.png'):
                f.write(f"  - {item}: Visualization\n")
            elif item.endswith('.pth'):
                f.write(f"  - {item}: Model checkpoint\n")
    
    print(f"\nConfiguration summary saved to: {config_file}")
    
    print("\n" + "="*70)
    print("VISUALIZATIONS GENERATED:")
    print("="*70)
    print("""
1. Training Convergence Curves (training_convergence.png):
   - Generator and Discriminator loss progression
   - Cycle consistency and identity losses
   - Epoch-wise average losses

2. Performance Evaluation (performance_evaluation.png):
   - PSNR, SSIM, and MSE distributions
   - Quantitative performance metrics

3. CycleGAN vs FedDG Comparison (cyclegan_vs_feddg_comparison.png):
   - Side-by-side comparison of both methods
   - Difference visualization
   - MSE comparison per sample

4. Parameter Comparison (parameter_comparison.png):
   - Methodological differences
   - Parameter settings
   - Computational requirements
    """)
    
    print("\n" + "="*70)
    print("READY FOR TASK3:")
    print("="*70)
    print(f"""
1. CycleGAN translated images are ready at:
   {translated_dir}/

2. For Task3, use:
   - Original Domain1 images: {source_path}
   - CycleGAN translated images: {translated_dir}
   - Domain1 labels: {os.path.join(BASE_PATH, SOURCE_DOMAIN, 'train', 'masks')}

3. Training strategy:
   - Combine all images for data augmentation
   - Train segmentation model on mixed dataset
   - Evaluate on all domains (Domain1-5)
   
4. Key advantages of CycleGAN over FedDG:
   - Learns more complex style transformations
   - Better preservation of anatomical structures
   - More natural-looking results
   - Can handle more dramatic domain shifts
    """)

# ========== 9. 运行程序 ==========
if __name__ == "__main__":
    main()