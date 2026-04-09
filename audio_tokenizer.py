
# #-------------------------------------------------modified..................

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# ------------------------
# Encoder / Decoder
# -------------------------

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=(1, 2)):
        """
        channels: number of channels
        dilation: tuple of dilation rates for the two conv layers
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=dilation[0], dilation=dilation[0]
            ),
            nn.LeakyReLU(0.2),
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=dilation[1], dilation=dilation[1]
            ),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=1, z_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            ResBlock(64),

            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            ResBlock(128),

            nn.Conv1d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            ResBlock(256),

            nn.Conv1d(256, z_dim, 4, 2, 1),  # final downsample
            nn.LeakyReLU(0.2)
        )

        # bottleneck 1×1 conv
        self.proj = nn.Conv1d(z_dim, z_dim, 1)

        # LayerNorm across channels
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, x):
        z = self.net(x)
        z = self.proj(z)

        # LayerNorm expects (B, T, C)
        z = z.permute(0, 2, 1)
        z = self.norm(z)
        z = z.permute(0, 2, 1)

        # L2 normalize → critical for stable quantization
        z = F.normalize(z, p=2, dim=1)

        return z  # (B, z_dim, T_down)



class Decoder(nn.Module):
    def __init__(self, out_channels=1, z_dim=256):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(z_dim, z_dim, 1),
            nn.LeakyReLU(0.2)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose1d(z_dim, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            ResBlock(256),

            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            ResBlock(128),

            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            ResBlock(64),

            nn.ConvTranspose1d(64, out_channels, 4, 2, 1)
        )

    def forward(self, q):
        q = self.initial(q)
        return self.net(q)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256,
                 commitment_cost=0.1, decay=0.99, eps=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Initialize embeddings
        embed = torch.randn(num_embeddings, embedding_dim) * 0.1
        self.register_buffer("embedding", embed)          # (K, D)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        """
        z: (B, D, T)
        """
        B, D, T = z.shape
        flat = z.permute(0, 2, 1).contiguous().view(-1, D)  # (B*T, D)

        # Compute distances (B*T, K)
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2 * flat @ self.embedding.t()
        )

        # Get nearest embedding
        encoding_indices = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat.dtype)

        quantized = encodings @ self.embedding       # (B*T, D)
        quantized = quantized.view(B, T, D).permute(0, 2, 1).contiguous()

        # EMA update
        if self.training:
            with torch.no_grad():
                cluster_sum = encodings.sum(0)        # (K,)
                embed_sum = encodings.t() @ flat      # (K, D)

                self.cluster_size.mul_(self.decay).add_(cluster_sum, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Normalize embeddings
                n = (self.cluster_size + self.eps).unsqueeze(1)
                self.embedding.copy_(self.embed_avg / n)
                self.embedding.copy_(F.normalize(self.embedding, dim=1))

        # Losses
        e_loss = F.mse_loss(z.detach(), quantized)
        q_loss = F.mse_loss(z, quantized.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        indices = encoding_indices.view(B, T)

        return quantized, vq_loss, indices


# ------------------------
# VQ-VAE wrapper
# -------------------------
class VQVAE(nn.Module):
    def __init__(self, z_dim=256, num_embeddings=1024):
        super().__init__()
        self.encoder = Encoder(1, z_dim)
        self.quantizer = VectorQuantizerEMA(num_embeddings, z_dim)
        self.decoder = Decoder(1, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(z)
        x_rec = self.decoder(quantized)
        return x_rec, vq_loss, indices


# ------------------------
# Eval helpers (no gradient)
# -------------------------
@torch.no_grad()
def tokenize_audio(model, waveform):
    model.eval()
    z = model.encoder(waveform)
    _, _, indices = model.quantizer(z)
    return indices.cpu().numpy()

@torch.no_grad()
def reconstruct_from_indices(model, indices):
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices).to(model.quantizer.embedding.device)
    
    B, T = indices.shape
    D = model.quantizer.embedding.shape[1]  # embedding_dim, not num_embeddings
    emb = model.quantizer.embedding          # (num_embeddings, embedding_dim)
    
    flat = emb[indices.view(-1)]             # (B*T, embedding_dim)
    q = flat.view(B, T, D).permute(0, 2, 1).contiguous()  # (B, D, T)
    
    x_rec = model.decoder(q)
    return x_rec


import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

# ------------------------
# Assume your VQVAE, Encoder, Decoder, VectorQuantizerEMA are already defined
# ------------------------

# Create dummy dataset: 10 audio samples, each length 1024
# B = 10      # batch size
# T = 1024    # audio length
# dummy_audio = torch.randn(B, 1, T)  # (B,1,T)

# # Wrap in a DataLoader
# dataset = TensorDataset(dummy_audio)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

# # Create model
# z_dim = 64           # smaller for testing
# num_embeddings = 32  # smaller for testing
# model = VQVAE(z_dim=z_dim, num_embeddings=num_embeddings)

# # Optimizer
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# # Move to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# # ------------------------
# # Single training step on dummy data
# # -----------------------
# for batch in dataloader:
#     x = batch[0].to(device)  # (B,1,T)
    
#     model.train()
#     optimizer.zero_grad()
    
#     x_rec, vq_loss, indices = model(x)
    
#     # Reconstruction loss (L1)
#     recon_loss = F.l1_loss(x_rec, x)
    
#     # Total loss
#     loss = recon_loss + vq_loss
#     print(loss)
#     # Backprop
#     loss.backward()
#     optimizer.step()
    
#     # print("Batch Loss:", loss.item())
#     # print("Reconstruction Loss:", recon_loss.item())
#     # print("VQ Loss:", vq_loss.item())
#     # print("Token shape:", indices.shape)


#--------------------------------------------------------

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, files, segment_length=16000):
        self.files = files
        self.segment_length = segment_length
        self.sr = 16000

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(dim=0)   # mono
        wav = wav / wav.abs().max()  # normalize
        
        # random crop
        if wav.shape[0] >= self.segment_length:
            start = torch.randint(0, wav.shape[0] - self.segment_length, (1,))
            wav = wav[start:start+self.segment_length]
        else:
            wav = F.pad(wav, (0, self.segment_length - wav.shape[0]))
        
        return wav.unsqueeze(0)

    def __len__(self):
        return len(self.files)
#-----------------------------------------------



import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
AUDIO_DIR = "/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/blues"

# collect audio file paths
audio_files = [
    os.path.join(AUDIO_DIR, f)
    for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith((".wav", ".mp3", ".flac"))
]
print("Found", len(audio_files), "audio files")

audio_files = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for f in files:
        if f.lower().endswith((".wav", ".mp3", ".flac")):
            audio_files.append(os.path.join(root, f))

dataset = AudioDataset(audio_files, segment_length=16000)

loader = DataLoader(dataset, batch_size=8, shuffle=True)


#---------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE(z_dim=512, num_embeddings=1024).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


#-------------------------------------------------


import torch
import torch.nn.functional as F

def multiscale_stft_loss(x, x_rec):
    """
    x, x_rec: (B, 1, T)
    Returns scalar STFT loss
    """
    losses = []
    scales = [
        (2048, 512, 2048),
        (1024, 256, 1024),
        (512, 128, 512),
    ]

    x = x[:, 0]       # (B, T)
    x_rec = x_rec[:, 0]

    for n_fft, hop, win in scales:
        window = torch.hann_window(win).to(x.device).detach()

        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True
        )

        Xr = torch.stft(
            x_rec,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True
        )

        losses.append((X - Xr).abs().mean())

    return sum(losses)

# --- parallel_wavegan/losses/stft_loss.py ---



# class SpectralConvergenceLoss(nn.Module):
#     def forward(self, x_mag, y_mag):
#         return torch.norm(y_mag - x_mag, p='fro') / torch.norm(x_mag, p='fro')


# class LogSTFTMagnitudeLoss(nn.Module):
#     def forward(self, x_mag, y_mag):
#         return F.l1_loss(torch.log(x_mag), torch.log(y_mag))


# def stft_fn(y, n_fft, hop_size, win_length, window):
#     window = window.to(y.device)  # move window to same device as input
#     spec = torch.stft(
#         y,
#         n_fft=n_fft,
#         hop_length=hop_size,
#         win_length=win_length,
#         window=window,
#         center=True,
#         pad_mode="reflect",
#         normalized=False,
#         onesided=True,
#         return_complex=True,
#     )
#     mag = torch.sqrt(torch.clamp(spec.abs() ** 2, min=1e-7))
#     return mag


# class STFTLoss(nn.Module):
#     def __init__(self, fft_size, hop_size, win_length, window_fn=torch.hann_window):
#         super().__init__()
#         self.fft_size = fft_size
#         self.hop_size = hop_size
#         self.win_length = win_length
#         self.window = window_fn(win_length)
#         self.sc_loss = SpectralConvergenceLoss()
#         self.mag_loss = LogSTFTMagnitudeLoss()

#     def forward(self, x, y):
#         x_mag = stft_fn(x, self.fft_size, self.hop_size, self.win_length, self.window)
#         y_mag = stft_fn(y, self.fft_size, self.hop_size, self.win_length, self.window)
#         sc = self.sc_loss(x_mag, y_mag)
#         mag = self.mag_loss(x_mag, y_mag)
#         return sc + mag


# class MultiResolutionSTFTLoss(nn.Module):
#     def __init__(self,
#                  fft_sizes=[1024, 2048, 512],
#                  hop_sizes=[256, 512, 128],
#                  win_lengths=[1024, 2048, 512],
#                  window_fn=torch.hann_window):
#         super().__init__()
#         self.losses = nn.ModuleList([
#             STFTLoss(fft, hop, win, window_fn)
#             for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
#         ])

#     def forward(self, x, y):
#         # x: (B,1,T) or (B,T)
#         if x.dim() == 3:
#             x = x[:,0]
#             y = y[:,0]
#         return sum(l(x, y) for l in self.losses)
# stft_loss_fn = MultiResolutionSTFTLoss().to(device)


#--------------------------------------------------


num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0.0
    total_l1 = 0.0
    total_stft = 0.0
    total_vq = 0.0

    for batch_idx, batch_waveform in enumerate(loader):

        x = batch_waveform.to(device)  # (B,1,T)
        optimizer.zero_grad()
        model.train()

        # --------------------
        # Forward pass
        # --------------------
        x_rec, vq_loss, indices = model(x)

        # Waveform L1 loss
        l1_loss = F.l1_loss(x_rec, x)

        # Multi-resolution STFT loss
        stft_loss = multiscale_stft_loss(x, x_rec)

        # Combined loss
        loss = l1_loss + 0.1 * stft_loss + vq_loss

        # --------------------
        # Backprop
        # --------------------
        loss.backward()
        optimizer.step()

        # Track stats
        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_stft += stft_loss.item()
        total_vq += vq_loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Batch [{batch_idx}/{len(loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"L1: {l1_loss.item():.4f} | "
                f"STFT: {stft_loss.item():.4f} | "
                f"VQ: {vq_loss.item():.4f}"
            )

    # End-of-epoch summary
    print(
        f"\nEPOCH {epoch+1} SUMMARY:\n"
        f"  Avg Loss : {total_loss/len(loader):.4f}\n"
        f"  Avg L1   : {total_l1/len(loader):.4f}\n"
        f"  Avg STFT : {total_stft/len(loader):.4f}\n"
        f"  Avg VQ   : {total_vq/len(loader):.4f}\n"
    )


#----------------------------------------------------

import torchaudio
import torch
import numpy as np

# pick a file from your dataset
AUDIO_PATH = "/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/blues/blues.00001.wav"

model.eval()

# ----------------------------------------------------
# 1. Load audio from dataset
# ----------------------------------------------------
waveform, sr = torchaudio.load(AUDIO_PATH)     # shape: (C, T)
# Make waveform ready for GPU and correct dtype
waveform = waveform.mean(dim=0, keepdim=True)  # mono
waveform = waveform.unsqueeze(0)               # (1,1,T)
waveform = waveform.float().to(device)         # <- ensure float + same device


print("Loaded waveform:", waveform.shape)

# ----------------------------------------------------
# 2. Convert audio → tokens
# ----------------------------------------------------
tokens = tokenize_audio(model, waveform)    # numpy array (1, T_down)
# When using indices for reconstruction
# tokens = tokens.long().to(device)

print("Token shape:", tokens.shape)

# ----------------------------------------------------
# 3. Reconstruct audio using ONLY indices
# ----------------------------------------------------
reconstructed = reconstruct_from_indices(model, tokens)
reconstructed = reconstructed.cpu().detach()

print("Reconstructed shape:", reconstructed.shape)

# ----------------------------------------------------
# 4. Save both audios
# ----------------------------------------------------
# torchaudio.save("original.wav", waveform[0].cpu(), sample_rate=sr)
torchaudio.save("reconstructed3.wav", reconstructed[0], sample_rate=sr)

print("Saved original.wav and reconstructed.wav!")


#--------------------------------

import kagglehub

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("Path to dataset files:", path)