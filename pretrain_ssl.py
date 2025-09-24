# ssl_train.py
import os, math, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_video
from tqdm import tqdm 

# =============== Dataset (unlabeled) =================
class UnlabeledVideoList(Dataset):
    """
    Mỗi dòng trong list_file là đường dẫn tuyệt đối đến 1 video (mp4/avi/...).
    Không cần nhãn. Trả về clip (C,T,H,W) với T frames đã được resize.
    """
    def __init__(self, list_file, num_frames=16, img_size=112):
        with open(list_file) as f:
            self.paths = [x.strip() for x in f if x.strip()]
        self.num_frames = num_frames
        self.img_size = img_size
        self.tf = T.Compose([
            T.ConvertImageDtype(torch.float32),  # [0,255] -> float
            T.Resize((img_size, img_size)),
        ])

    def _sample_indices(self, total):
        # uniform hoặc random segment, đảm bảo đủ num_frames
        if total <= self.num_frames:
            idx = np.linspace(0, total-1, self.num_frames).astype(int)
        else:
            start = random.randint(0, total - self.num_frames)
            idx = np.arange(start, start + self.num_frames)
        return idx

    def __getitem__(self, idx):
        path = os.path.normpath(self.paths[idx])
        # path = self.paths[idx]
        # read_video trả (T,H,W,C) trong range [0,255]
        frames, _, _ = read_video(path, pts_unit="sec")  # torch.uint8
        T_total = frames.shape[0]
        sel = self._sample_indices(T_total)
        clip = frames[sel]  # (T,H,W,C)
        clip = clip.permute(0,3,1,2)  # (T,C,H,W)
        # áp dụng transform frame-wise rồi ghép thành (C,T,H,W)
        clip = torch.stack([self.tf(f) for f in clip], dim=0)  # (T,C,H,W)
        clip = clip.permute(1,0,2,3).contiguous()  # (C,T,H,W)
        return clip

    def __len__(self):
        return len(self.paths)

# =============== MAE building blocks =================
class PatchEmbed2D(nn.Module):
    """Patchify từng frame (2D) rồi xếp theo thời gian: seq_len = T * Npatch."""
    def __init__(self, in_ch=3, embed_dim=384, patch=16):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):  # x: (B,C,T,H,W)
        B,C,T,H,W = x.shape
        x = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)        # (B*T,C,H,W)
        x = self.proj(x)                                       # (B*T,D,H/ps,W/ps)
        x = x.flatten(2).transpose(1,2)                        # (B*T, N, D)
        N = x.shape[1]
        x = x.reshape(B, T, N, -1).flatten(1,2)                # (B, L=T*N, D)
        return x, N

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, dim, depth=12, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm   = nn.LayerNorm(dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class VideoMAE_SSL(nn.Module):
    """
    MAE tối giản cho video:
    - Patchify từng frame 2D (tubelet 1x), sequence = T*Npatch.
    - Mask 1 - keep_ratio theo vị trí sequence.
    - Encoder ViT trên tokens đã keep.
    - Decoder nhẹ nhận (enc_tokens + mask_tokens) để reconstruct.
    """
    def __init__(self, img_size=112, patch=16, dim=384, depth=12, heads=6,
                 mlp_ratio=4.0, dec_dim=192, dec_depth=4, mask_ratio=0.75):
        super().__init__()
        self.embed = PatchEmbed2D(embed_dim=dim, patch=patch)
        self.mask_ratio = mask_ratio

        # Encoder
        self.pos_enc = nn.Parameter(torch.zeros(1, 1, dim))  # sẽ resize động
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.encoder = SimpleTransformer(dim, depth, heads, mlp_ratio)

        # Decoder
        self.dec_proj = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1,1,dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = SimpleTransformer(dec_dim, dec_depth, heads, mlp_ratio)
        self.head = nn.Linear(dec_dim, 3*patch*patch)  # dự đoán pixel của 1 patch (per-frame)

        self.patch = patch
        self.dim   = dim
        self.dec_dim = dec_dim

    def _pos_embed(self, L, D, device):
        # sinusoidal pos embedding động theo độ dài L, chiều D
        pe = torch.arange(L, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device) * (-math.log(10000.0)/D))
        pos = torch.zeros(1, L, D, device=device)
        pos[:,:,0::2] = torch.sin(pe*div)
        pos[:,:,1::2] = torch.cos(pe*div)
        return pos

    def forward(self, x):  # x: (B,C,T,H,W)
        B = x.shape[0]
        tokens, n_per_frame = self.embed(x)           # (B, L, D)
        L, D = tokens.size(1), tokens.size(2)
        pos  = self._pos_embed(L, D, x.device)

        # ---- random mask ----
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)             # asc
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1,-1,D))
        pos_vis = torch.gather(pos.expand(B,-1,-1), 1, ids_keep.unsqueeze(-1).expand(-1,-1,D))

        # ---- encoder ----
        enc = self.encoder(x_vis + pos_vis)                   # (B, len_keep, D)

        # ---- decoder ----
        dec_tokens = self.dec_proj(enc)
        mask_tokens = self.mask_token.expand(B, L-len_keep, -1)
        dec_ = torch.cat([dec_tokens, mask_tokens], dim=1)    # (B, L, dec_dim) in shuffled order
        dec_ = torch.gather(dec_, 1, ids_restore.unsqueeze(-1).expand(-1,-1,self.dec_dim))
        dec_ = self.decoder(dec_)
        pred = self.head(dec_)                                 # (B, L, 3*ps*ps)
        return pred, ids_restore, n_per_frame

def mae_recon_loss(pred, target, ids_restore, patch=16):
    """
    pred: (B, L, 3*ps*ps) — dự đoán tất cả vị trí (cả keep + mask)
    chỉ tính MSE trên các vị trí bị mask (chuẩn MAE)
    """
    B,C,T,H,W = target.shape
    # patchify target
    tgt = target.permute(0,2,1,3,4).reshape(B*T, C, H, W)  # (B*T,C,H,W)
    patches = torch.nn.functional.unfold(tgt, kernel_size=patch, stride=patch)  # (B*T, 3*ps*ps, N)
    patches = patches.permute(0,2,1)                        # (B*T, N, 3*ps*ps)
    patches = patches.reshape(B, -1, patches.shape[-1])     # (B, L=T*N, 3*ps*ps)

    L = patches.size(1)
    len_keep = pred.size(1) - (L - int(L*(1 - 0.75)))  # ước lượng; hoặc truyền mask trực tiếp nếu bạn giữ lại
    mask = torch.ones(B, L, device=pred.device)
    mask[:, :len_keep] = 0.
    mask = torch.gather(mask, 1, ids_restore)  # 1 với vị trí mask

    loss = ((pred - patches) ** 2).mean(dim=-1)            # (B, L)
    loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
    return loss

# =============== Train Loop =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_list', type=str, required=True, help='txt: mỗi dòng là đường dẫn video')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_frames', type=int, default=16)
    ap.add_argument('--img_size', type=int, default=112)
    ap.add_argument('--mask_ratio', type=float, default=0.75)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--out', type=str, default='./checkpoints/ssl_ssv2_vits')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = UnlabeledVideoList(args.train_list, args.num_frames, args.img_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=True)

    model = VideoMAE_SSL(
        img_size=args.img_size, patch=16, dim=384, depth=12, heads=6,
        dec_dim=192, dec_depth=4, mask_ratio=args.mask_ratio
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        t0 = time.time()
        for clip in tqdm(dl, desc=f"Epoch {ep+1}/{args.epochs}", ncols=100):
            clip = clip.to(device)          # (B,C,T,H,W), float in [0,1]
            pred, ids_restore, npf = model(clip)
            loss = mae_recon_loss(pred, clip, ids_restore, patch=16)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()

        sched.step()
        avg_loss = total/len(dl)
        print(f"[{ep+1:03d}/{args.epochs}] SSL loss = {avg_loss:.4f} | {(time.time()-t0)/60:.1f} min")

        # Lưu checkpoint đầy đủ
        if (ep+1) % 50 == 0 or ep+1 == args.epochs:
            ckpt = {
                'epoch': ep+1,
                'model': model.state_dict(),         # full model
                'module': model.state_dict(),        # alias (hợp với code cũ)
                'encoder_state': {                   # chỉ encoder
                    k.replace('encoder.', ''): v
                    for k,v in model.state_dict().items()
                    if k.startswith('encoder.')
                },
                'optimizer': optim.state_dict(),     # optimizer state
                'scheduler': sched.state_dict(),     # scheduler state
            }
            path = os.path.join(args.out, f"epoch_{ep+1}.pth")
            torch.save(ckpt, path)
            print(f"Saved checkpoint: {path}")

    # Ngoài ra, lưu riêng encoder-only để nạp ở pha SL:
    enc_path = os.path.join(args.out, "encoder_only_final.pth")
    torch.save({'epoch': args.epochs,
                'model': model.state_dict(),
                'module': model.state_dict(),
                'encoder_state': {k.replace('encoder.', ''): v
                                  for k,v in model.state_dict().items()
                                  if k.startswith('encoder.')}}, enc_path),
    print(f"Saved encoder-only: {enc_path}")

if __name__ == "__main__":
    main()
