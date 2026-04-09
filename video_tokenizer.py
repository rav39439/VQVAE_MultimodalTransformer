import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -----------------------------
# Utility: 3D Residual Block
# -----------------------------
class ResBlock3D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

# -----------------------------
# Encoder / Decoder (3D)
# -----------------------------
class VideoEncoder(nn.Module):
    def __init__(self, in_ch=3, z_dim=256):
        super().__init__()
        # downsample spatial & temporal by factor 2 per stage
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=4, stride=(1,2,2), padding=(1,1,1)),   # keep T, half H/W
            nn.LeakyReLU(0.2),
            ResBlock3D(64),

            nn.Conv3d(64, 128, kernel_size=4, stride=(2,2,2), padding=1),            # half T,H,W
            nn.LeakyReLU(0.2),
            ResBlock3D(128),

            nn.Conv3d(128, 256, kernel_size=4, stride=(2,2,2), padding=1),
            nn.LeakyReLU(0.2),
            ResBlock3D(256),

            nn.Conv3d(256, z_dim, kernel_size=3, stride=1, padding=1),               # final channels = z_dim
            nn.LeakyReLU(0.2),
        )

        # projection & normalization across channels for stability
        self.proj = nn.Conv3d(z_dim, z_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, z_dim)  # more stable than LayerNorm for conv feature maps

    def forward(self, x):
        # expected x: (B, C, T, H, W)
        z = self.net(x)
        z = self.proj(z)
        # GroupNorm expects (B, C, D, T, H) so it's aligned already
        z = self.norm(z)
        # return (B, D, T', H', W')
        return z

class VideoDecoder(nn.Module):
    def __init__(self, out_ch=3, z_dim=256):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv3d(z_dim, z_dim, kernel_size=1),
            nn.LeakyReLU(0.2),
        )

        self.net = nn.Sequential(
            # upsample 2x T,H,W except first stage we used (1,2,2) so invert that
            nn.ConvTranspose3d(z_dim, 256, kernel_size=4, stride=(1,2,2), padding=(1,1,1)),
            nn.LeakyReLU(0.2),
            ResBlock3D(256),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=(2,2,2), padding=1),
            nn.LeakyReLU(0.2),
            ResBlock3D(128),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=(2,2,2), padding=1),
            nn.LeakyReLU(0.2),
            ResBlock3D(64),

            nn.Conv3d(64, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, q):
        q = self.init(q)
        return self.net(q)  # (B, C, T, H, W)

# -----------------------------
# VectorQuantizerEMA (optimized)
# -----------------------------
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256,
                 commitment_cost=0.25, decay=0.999, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim) * 0.01
        self.register_buffer("embedding", embed)            # (K, D)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        # z: (B, D, T, H, W)
        B, D, T, H, W = z.shape
        flat = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, D)  # (B*T*H*W, D)

        # distances
        emb = self.embedding                                 # (K, D)
        # compute squared distances efficiently
        dist = (flat.pow(2).sum(dim=1, keepdim=True)
                + emb.pow(2).sum(dim=1)
                - 2.0 * flat @ emb.t())                     # (N, K) with N=B*T*H*W

        encoding_indices = torch.argmin(dist, dim=1)         # (N,)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)  # (N, K)

        # quantized vectors
        quantized_flat = encodings @ emb                     # (N, D)
        quantized = quantized_flat.view(B, T, H, W, D).permute(0, 4, 1, 2, 3).contiguous()  # (B,D,T,H,W)

        # EMA updates
        if self.training:
            with torch.no_grad():
                cluster_sum = encodings.sum(dim=0)          # (K,)
                embed_sum = encodings.t() @ flat            # (K, D)

                self.cluster_size.mul_(self.decay).add_(cluster_sum, alpha=1.0 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

                n = (self.cluster_size + self.eps).unsqueeze(1)
                new_embed = self.embed_avg / n
                # normalize embedding vectors to avoid blowup
                new_embed = F.normalize(new_embed, dim=1)
                self.embedding.copy_(new_embed)

        # losses
        e_loss = F.mse_loss(z.detach(), quantized)
        q_loss = F.mse_loss(z, quantized.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss

        # straight-through
        quantized_st = z + (quantized - z).detach()

        indices = encoding_indices.view(B, T, H, W)  # long tensor

        return quantized_st, vq_loss, indices

# -----------------------------
# VQ-VAE Video wrapper
# -----------------------------
class VideoVQVAE(nn.Module):
    def __init__(self, in_ch=3, z_dim=256, num_embeddings=1024):
        super().__init__()
        self.encoder = VideoEncoder(in_ch, z_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings, z_dim)
        self.decoder = VideoDecoder(in_ch, z_dim)

    def forward(self, x):
        z = self.encoder(x)                 # (B, D, T', H', W')
        q, vq_loss, indices = self.quantizer(z)
        x_rec = self.decoder(q)
        return x_rec, vq_loss, indices

# -----------------------------
# Tokenize / Reconstruct helpers
# -----------------------------
@torch.no_grad()
def tokenize_video(model: VideoVQVAE, video: torch.Tensor) -> torch.LongTensor:
    """
    video: (B, C, T, H, W) on same device as model
    returns indices: (B, T', H', W') long on model device
    """
    model.eval()
    z = model.encoder(video)
    _, _, indices = model.quantizer(z)
    return indices.long()

@torch.no_grad()
def reconstruct_from_indices(model: VideoVQVAE, indices: torch.LongTensor) -> torch.Tensor:
    """
    indices: (B, T', H', W') on same device as model
    returns reconstructed video: (B, C, T, H, W)
    """
    if indices.dtype != torch.long:
        indices = indices.long()
    # get embedding table and shape
    emb = model.quantizer.embedding            # (K, D)
    B, T_, H_, W_ = indices.shape
    D = emb.shape[1]

    flat = emb[indices.view(-1)]               # (B*T'*H'*W', D)
    q = flat.view(B, T_, H_, W_, D).permute(0, 4, 1, 2, 3).contiguous()  # (B,D,T',H',W')

    x_rec = model.decoder(q)
    return x_rec

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoVQVAE(in_ch=3, z_dim=128, num_embeddings=512).to(device)

# dummy batch: B=2, C=3, T=16, H=64, W=64
x = torch.randn(2, 3, 16, 64, 64, device=device)

# forward
x_rec, vq_loss, indices = model(x)
print("x_rec", x_rec.shape)             # (2,3,16,64,64)
print("indices", indices.shape)         # (2, T', H', W') e.g. (2,4,8,8)

# tokenize then reconstruct
tokens = tokenize_video(model, x)
xr = reconstruct_from_indices(model, tokens)
print("recon from tokens", xr.shape)






#------------------------------------------------

import kagglehub

# Download latest version
path = kagglehub.dataset_download("hasaniut/bdslw60videoclips")

print("Path to dataset files:", path)

#----------------------------------------------------


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import decord
from decord import VideoReader, cpu
import numpy as np
import cv2

class VideoDataset(Dataset):
    def __init__(self, root="/kaggle/input/bdslw60videoclips/aam", num_frames=16, resolution=128):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith((".mp4", ".avi", ".mov"))]

        self.num_frames = num_frames
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def load_video(self, path):
        vr = VideoReader(path, ctx=cpu())
        total_frames = len(vr)

        # --- Uniform Frame Selection ---
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        else:
            # pad by repeating last frame
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)

        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

        # Resize & normalize
        resized = []
        for f in frames:
            f = cv2.resize(f, (self.resolution, self.resolution))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized.append(f)

        frames = np.stack(resized)  # (T, H, W, 3)

        # Convert → Tensor (C, T, H, W)
        frames = torch.tensor(frames).float() / 255.0
        frames = frames.permute(3, 0, 1, 2)  # (3, T, H, W)
        frames = (frames * 2) - 1  # → [-1, 1]

        return frames

    def __getitem__(self, idx):
        video_path = self.files[idx]
        video_tensor = self.load_video(video_path)
        return video_tensor


#--------------------------------------------------

dataset = VideoDataset(
    root="/kaggle/input/bdslw60videoclips/aam",
    num_frames=16,       # number of frames per clip
    resolution=256       # resize to 128×128
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"


#--------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# for file loading
import os
import numpy as np

# if your dataset uses torchvision transforms
import torchvision.transforms as T

# -----------------------------------
# MAKE SURE device IS DEFINED
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------------
# DEFINE num_embeddings
# -----------------------------------
NUM_EMBEDDINGS = 512   # you can change this

# -----------------------------------
# CREATE MODEL
# -----------------------------------
model = VideoVQVAE(
    in_ch=3,
    z_dim=256,
    num_embeddings=NUM_EMBEDDINGS,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-4)

# L1 reconstruction loss works best for VQ-VAE
reconstruction_loss = nn.L1Loss()


# -----------------------------------------------------------
#  TRAINING LOOP
# -----------------------------------------------------------

epochs = 60

for epoch in range(epochs):
    for batch in loader:
        # batch: (B, C, T, H, W)
        batch = batch.to(device)

        optimizer.zero_grad()

        # forward pass through VQ-VAE
        x_rec, vq_loss, _ = model(batch)

        # reconstruction loss + VQ loss
        rec_loss = reconstruction_loss(x_rec, batch)
        loss = rec_loss + vq_loss

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}  |  loss={loss.item():.4f}  rec={rec_loss.item():.4f}")


# -----------------------------------------------------------
#  Test tokenization + reconstruction
# -----------------------------------------------------------

sample = next(iter(loader))[:1].to(device)   # pick 1 video

tokens = tokenize_video(model, sample)
recon = reconstruct_from_indices(model, tokens)

print("Tokenized shape:", tokens.shape)
print("Reconstructed video:", recon.shape)


#------------------------------------------------


import imageio
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# -------------------------------------------------------
# LOAD FRAMES FROM A VIDEO FILE
# -------------------------------------------------------
def load_demo_video_from_mp4(video_path, max_frames=8, size=64, device="cuda"):
    reader = imageio.get_reader(video_path)

    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])

    frames = []
    
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        img = Image.fromarray(frame).convert("RGB")
        frames.append(transform(img))

    reader.close()

    if len(frames) == 0:
        raise ValueError("❌ No frames loaded from video")

    video = torch.stack(frames, 0)           # (T, C, H, W)
    video = video.permute(1, 0, 2, 3)        # (C, T, H, W)
    video = video.unsqueeze(0)               # (1, C, T, H, W)

    print("Loaded video shape:", video.shape)
    return video.to(device), frames          # return original frames!


# -------------------------------------------------------
# SAVE VIDEO USING PROPER MP4 WRITER
# -------------------------------------------------------
def save_video_mp4(video_tensor, path="output.mp4", fps=8):
    video = video_tensor.squeeze(0).permute(1,0,2,3)  # (T,C,H,W)

    # ---- IMPORTANT FIX: scale output from [-1,1] → [0,1] ----
    video = (video + 1) / 2
    video = video.clamp(0, 1)

    with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
        for t in range(video.shape[0]):
            frame = video[t].permute(1,2,0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

    print(f"✅ Saved: {path}")


# -------------------------------------------------------
# SAVE ORIGINAL VIDEO FOR COMPARISON
# -------------------------------------------------------
def save_original_video(original_frames, path="original.mp4", fps=8):
    with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
        for frame in original_frames:
            writer.append_data((frame.permute(1,2,0).numpy() * 255).astype(np.uint8))
    print("✅ Saved original video:", path)


# -------------------------------------------------------
# LOAD VIDEO
# -------------------------------------------------------
video_path = "/kaggle/input/bdslw60videoclips/aam/U10W37F_trial_0_L.mp4"

demo_video, original_frames = load_demo_video_from_mp4(
    video_path,
    max_frames=8,
    size=512,
    device=device
)

# -------------------------------------------------------
# SAVE ORIGINAL VIDEO
# -------------------------------------------------------
save_original_video(original_frames, "original1.mp4", fps=8)


# -------------------------------------------------------
# RUN THROUGH MODEL
# -------------------------------------------------------
model.eval()
with torch.no_grad():
    x_rec, vq_loss, indices = model(demo_video)

print("Reconstructed:", x_rec.shape)
print("Token indices:", indices.shape)

# -------------------------------------------------------
# SAVE RECONSTRUCTION
# -------------------------------------------------------
save_video_mp4(x_rec, "reconstructed11.mp4", fps=8)

# -------------------------------------------------------
# TOKENIZE → RECONSTRUCT AGAIN
# -------------------------------------------------------
tokens = tokenize_video(model, demo_video)
recon_from_tokens = reconstruct_from_indices(model, tokens)

save_video_mp4(recon_from_tokens, "reconstructed_from_tokens11.mp4", fps=8)

print("Tokens:", tokens.shape)



#-------------------------------------------------


