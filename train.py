import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
from cleanfid import fid
from sklearn.decomposition import PCA
import lpips
import numpy as np


def compute_lpips_score(fake_dir, device, num_pairs=5000):
    """
    Fake 이미지들끼리 LPIPS를 계산하여 생성된 이미지의 다양성을 측정

    Args:
        fake_dir: 생성된 이미지들이 저장된 디렉토리
        device: GPU/CPU 디바이스
        num_pairs: 비교할 이미지 쌍의 개수

    Returns:
        평균 LPIPS score (높을수록 더 다양함)
    """
    import os
    import random
    from torchvision.io import read_image

    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    fake_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".png")]



    total_lpips = 0.0
    count = 0

    for _ in range(min(num_pairs, len(fake_paths) * (len(fake_paths) - 1) // 2)):
        img1_path, img2_path = random.sample(fake_paths, 2)

        try:
            img1 = read_image(img1_path).float().to(device) / 255.0 * 2 - 1
            img2 = read_image(img2_path).float().to(device) / 255.0 * 2 - 1

            img1 = img1.unsqueeze(0)  # (1, 3, H, W)
            img2 = img2.unsqueeze(0)

            d = loss_fn(img1, img2)
            total_lpips += d.item()
            count += 1

        except Exception as e:
            print(f"Error processing images {img1_path}, {img2_path}: {e}")
            continue

    return total_lpips / count if count > 0 else None


def plot_single_metric(metric_scores, metric_name, save_path):
    epochs = [ep for ep, _ in metric_scores]
    values = [score for _, score in metric_scores]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_features_pca_centroid(real_features, pca_centroid, fake_features=None,
                                    save_path="/content/pca_feature_space_with_centroid.png"):
    """
    PCA 공간에서 real features, fake features, centroid를 시각화
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    plt.scatter(real_features[:, 0], real_features[:, 1], s=10, alpha=0.5, label='Real Features', c='blue')

    if fake_features is not None:
        plt.scatter(fake_features[:, 0], fake_features[:, 1], s=10, alpha=0.5, label='Fake Features', c='green')

    plt.scatter(pca_centroid[0], pca_centroid[1], s=300, c='red', marker='*', label='PCA Centroid', edgecolors='black',
                linewidth=2)

    plt.legend()
    plt.title("PCA Feature Space with Centroid")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def pca_centroid_penalty_loss(fake_img, pca_model, pca_centroid, avg_distance, vgg_extractor, penalty_scale=1.0):
    """
    PCA 공간에서 centroid와의 거리가 평균 거리를 넘어서면 패널티를 주는 loss

    Args:
        fake_img: (B, 3, H, W) 생성된 이미지
        pca_model: 학습된 PCA 모델
        pca_centroid: PCA 공간에서의 centroid (2D)
        avg_distance: 학습 데이터의 centroid와의 평균 거리
        vgg_extractor: VGG feature extractor
        penalty_scale: 패널티 스케일 조정 팩터
    """
    with torch.no_grad():
        vgg_extractor.eval()

    features = vgg_extractor(fake_img)  # (B, C, H, W)
    B, C, H, W = features.shape
    fake_vecs = features.view(B, C, -1).mean(dim=2)  # (B, C) - spatial average pooling

    fake_vecs_cpu = fake_vecs.detach().cpu().numpy()
    fake_pca = pca_model.transform(fake_vecs_cpu)  # (B, 2)

    fake_pca_tensor = torch.tensor(fake_pca, dtype=torch.float32, device=fake_img.device)
    pca_centroid_tensor = torch.tensor(pca_centroid, dtype=torch.float32, device=fake_img.device)

    distances = torch.norm(fake_pca_tensor - pca_centroid_tensor, dim=1)  # (B,)

    avg_distance_tensor = torch.tensor(avg_distance, dtype=torch.float32, device=fake_img.device)
    penalty = F.relu(distances - avg_distance_tensor)  # 평균 거리를 넘어서는 부분만 패널티

    return penalty_scale * penalty.mean()


def compute_pca_centroid_and_avg_distance(dataloader, vgg_extractor, device, n_components=2):

    vgg_extractor.eval()
    feature_list = []

    print("Extracting VGG features from training data...")
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i % 50 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
            imgs = data[0].to(device)
            feats = vgg_extractor((imgs + 1) / 2)
            B, C, H, W = feats.shape
            feats = feats.view(B, C, -1).mean(dim=2)  # (B, C) - spatial average pooling
            feature_list.append(feats.cpu())

    all_feats = torch.cat(feature_list, dim=0).numpy()  # (N, D)
    print(f"Total features extracted: {all_feats.shape}")

    print(f"Applying PCA with {n_components} components...")
    pca_model = PCA(n_components=n_components, random_state=42)
    real_pca_features = pca_model.fit_transform(all_feats)  # (N, n_components)

    pca_centroid = np.mean(real_pca_features, axis=0)  # (n_components,)

    distances = np.linalg.norm(real_pca_features - pca_centroid, axis=1)  # (N,)
    avg_distance = np.mean(distances)

    print(f"PCA Centroid: {pca_centroid}")
    print(f"Average distance to centroid: {avg_distance:.4f}")
    print(f"Distance std: {np.std(distances):.4f}")

    analysis_path = "/content/pca_centroid_analysis.txt"
    with open(analysis_path, 'w') as f:
        f.write(f"[PCA Centroid Analysis]\n")
        f.write(f"Dataset features shape: {all_feats.shape}\n")
        f.write(f"PCA components: {n_components}\n")
        f.write(f"Explained variance ratio: {pca_model.explained_variance_ratio_}\n")
        f.write(f"Total explained variance: {np.sum(pca_model.explained_variance_ratio_):.4f}\n")
        f.write(f"PCA Centroid: {pca_centroid}\n")
        f.write(f"Average distance to centroid: {avg_distance:.4f}\n")
        f.write(f"Distance statistics:\n")
        f.write(f"  Min: {np.min(distances):.4f}\n")
        f.write(f"  Max: {np.max(distances):.4f}\n")
        f.write(f"  Std: {np.std(distances):.4f}\n")
        f.write(f"  Median: {np.median(distances):.4f}\n")

    return pca_model, pca_centroid, avg_distance, real_pca_features


def train_gan(epoch_setting, dataset_name):
    print(f"\n{'=' * 50}")
    print(f"에폭 {epoch_setting}으로 학습 시작")
    print(f"{'=' * 50}\n")

    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    if dataset_name == "lsun_bedroom":
        dataroot = "/content/data/lsun_bedroom/data0/lsun/bedroom"
    elif dataset_name == "celebA":
        dataroot = "/content/data/celeba"
    elif dataset_name == "ffhq":
        dataroot = "/content/data/ffhq"
    elif dataset_name == "ImageNet64":
        dataroot = "/content/data/ImageNet64"

    workers = 8

    batch_size = 128

    image_size = 64

    nc = 3

    nz = 100

    ngf = 64

    ndf = 64

    num_epochs = epoch_setting

    lr_D = 0.0002
    lr_G = 0.0002
    beta1 = 0.5

    ngpu = 1

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator Code
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. ``(ngf*8) x 4 x 4``
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. ``(ngf*4) x 8 x 8``
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. ``(ngf*2) x 16 x 16``
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. ``(ngf) x 32 x 32``
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input):
            return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input).view(-1)

    class VGGFeatureExtractor(nn.Module):
        def __init__(self, layer_index=16):
            super().__init__()
            vgg = models.vgg16(pretrained=True).features[:layer_index + 1]
            self.features = vgg.eval()
            for param in self.features.parameters():
                param.requires_grad = False

        def forward(self, x):
            return self.features(x)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    vgg_extractor = VGGFeatureExtractor().to(device)

    print("\n" + "=" * 50)
    print("Computing PCA centroid and average distance...")
    print("=" * 50)
    pca_model, pca_centroid, avg_distance, real_pca_features = compute_pca_centroid_and_avg_distance(
        dataloader, vgg_extractor, device, n_components=2
    )

    visualize_features_pca_centroid(
        real_features=real_pca_features,
        pca_centroid=pca_centroid,
        fake_features=None,
        save_path="/content/initial_pca_space.png"
    )

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

    d_losses = []
    g_losses = []
    pca_penalty_losses = []
    fid_scores = []
    lpips_scores = []

    print("\nStarting Training Loop...")
    print("=" * 50)
    for epoch in range(num_epochs):

        for i, data in enumerate(dataloader, 0):


            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output_real = netD(real_cpu)
            errD_real = criterion(output_real, label_real)
            D_x = output_real.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_img = netG(noise)

            output_fake = netD(fake_img.detach())
            errD_fake = criterion(output_fake, label_fake)
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()


            netG.zero_grad()

            output_fake_G = netD(fake_img)
            errG_adv = criterion(output_fake_G, label_real)

            lambda_pca = 0.1
            errG_pca_penalty = pca_centroid_penalty_loss(
                fake_img, pca_model, pca_centroid, avg_distance, vgg_extractor, penalty_scale=1.0
            )

            errG = errG_adv + lambda_pca * errG_pca_penalty
            errG.backward()
            D_G_z2 = output_fake_G.mean().item()
            optimizerG.step()


            if i % 50 == 0:
                with torch.no_grad():
                    d_losses.append(errD.item())
                    g_losses.append(errG.item())
                    pca_penalty_losses.append(errG_pca_penalty.item())
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_adv: %.4f\tLoss_PCA: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errD.item(), errG_adv.item(), errG_pca_penalty.item(), D_x, D_G_z1, D_G_z2))

        if (epoch + 1) % 10 == 0:
            # Create two subplots side by side in one PNG file
            plt.figure(figsize=(12, 5))

            # First subplot - Adversarial losses
            plt.subplot(1, 2, 1)
            plt.title(f"Adversarial Losses (Epoch {epoch + 1})")
            plt.plot(g_losses, label="Generator")
            plt.plot(d_losses, label="Discriminator")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()

            # Second subplot - PCA penalty loss
            plt.subplot(1, 2, 2)
            plt.title(f"PCA Penalty Loss (Epoch {epoch + 1})")
            plt.plot(pca_penalty_losses, label="PCA Penalty Loss", color="orange")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()

            plt.tight_layout()
            plt.savefig(f"/content/combined_losses_epoch_{epoch + 1}.png")
            plt.close()

            with torch.no_grad():
                sample_noise = torch.randn(64, nz, 1, 1, device=device)
                sample_images = netG(sample_noise)
                vutils.save_image(sample_images.detach(),
                                  f"/content/samples_epoch_{epoch + 1}.png",
                                  normalize=True,
                                  nrow=8)

        if (epoch + 1) % 10 == 0:
            num_fake_samples = min(5000, len(real_pca_features))
            fake_features_list = []

            with torch.no_grad():
                num_batches = (num_fake_samples + batch_size - 1) // batch_size
                for _ in range(num_batches):
                    current_batch = min(batch_size, num_fake_samples - len(fake_features_list) * batch_size)
                    if current_batch <= 0:
                        break

                    noise = torch.randn(current_batch, nz, 1, 1, device=device)
                    fake_batch = netG(noise)

                    feats = vgg_extractor((fake_batch + 1) / 2)
                    B, C, H, W = feats.shape
                    vecs = feats.view(B, C, -1).mean(dim=2)  # (B, C)

                    vecs_cpu = vecs.cpu().numpy()
                    fake_pca = pca_model.transform(vecs_cpu)
                    fake_features_list.append(fake_pca)

                full_fake_pca_features = np.concatenate(fake_features_list, axis=0)

                subset_real = real_pca_features[:num_fake_samples]
                visualize_features_pca_centroid(
                    real_features=subset_real,
                    pca_centroid=pca_centroid,
                    fake_features=full_fake_pca_features,
                    save_path=f"/content/pca_space_with_generated_epoch_{epoch + 1}.png"
                )

        if (epoch + 1) % 10 == 0:
            real_dir = f"/content/GAN_epoch_{epoch + 1}_real_images"
            fake_dir = f"/content/GAN_epoch_{epoch + 1}_fake_images"
            os.makedirs(real_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)

            num_samples = 50000
            num_full_batches = num_samples // batch_size
            remaining_samples = num_samples % batch_size

            real_counter = 0
            fake_counter = 0

            with torch.no_grad():
                data_iter = iter(dataloader)
                for i in range(num_full_batches):
                    try:
                        real_batch = next(data_iter)[0].to(device)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        real_batch = next(data_iter)[0].to(device)

                    for j in range(real_batch.size(0)):
                        vutils.save_image(real_batch[j], f"{real_dir}/img_{real_counter}.png", normalize=True)
                        real_counter += 1

                    noise = torch.randn(batch_size, nz, 1, 1, device=device)
                    fake_batch = netG(noise)
                    for j in range(fake_batch.size(0)):
                        vutils.save_image(fake_batch[j], f"{fake_dir}/img_{fake_counter}.png", normalize=True)
                        fake_counter += 1

                if remaining_samples > 0:
                    try:
                        real_batch = next(data_iter)[0][:remaining_samples].to(device)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        real_batch = next(data_iter)[0][:remaining_samples].to(device)

                    for j in range(real_batch.size(0)):
                        vutils.save_image(real_batch[j], f"{real_dir}/img_{real_counter}.png", normalize=True)
                        real_counter += 1

                    noise = torch.randn(remaining_samples, nz, 1, 1, device=device)
                    fake_batch = netG(noise)
                    for j in range(fake_batch.size(0)):
                        vutils.save_image(fake_batch[j], f"{fake_dir}/img_{fake_counter}.png", normalize=True)
                        fake_counter += 1

            fid_score = fid.compute_fid(real_dir, fake_dir, mode="clean", dataset_res=64, num_workers=0)
            lpips_score = compute_lpips_score(fake_dir, device, num_pairs=5000)

            fid_scores.append((epoch + 1, fid_score))
            lpips_scores.append((epoch + 1, lpips_score))

            score_path = "/content/fid_lpips_scores.txt"
            if not os.path.exists(score_path):
                with open(score_path, 'w') as f:
                    f.write("Epoch\tFID\tLPIPS\n")
            with open(score_path, 'a') as f:
                f.write(f"{epoch + 1}\t{fid_score:.4f}\t{lpips_score:.4f}\n")

            plot_single_metric(fid_scores, "FID", "/content/fid_plot.png")
            plot_single_metric(lpips_scores, "LPIPS", "/content/lpips_plot.png")

    print("\nTraining completed!")


