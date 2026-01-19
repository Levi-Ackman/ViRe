import io
import numpy as np
import torch
import torch.nn as nn
import open_clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class GenVLMEmb(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_large_d",
        pretrained_path: str = "/data/gqyu/alib/weight/clip_vitb32_openai/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg.bin",
        device: str = "cuda:0",
        fs: int = 256,                    
        per_panel_height: int = 120,      
        panel_width: int = 1400,          
        dpi: int = 50,                   
        line_px: float = 6.0,             
    ):
        super().__init__()
        self.device = device
        self.fs = fs
        self.per_panel_height = per_panel_height
        self.panel_width = panel_width
        self.dpi = dpi
        self.line_px = float(line_px)
        
        self.model, self.image_preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_path or "laion2b_s26b_b102k_augreg", device=device
        )
        self.model.eval()

    @torch.no_grad()
    def generate_embeddings(self, in_data: torch.Tensor) -> torch.Tensor:
        assert in_data.ndim == 3, "in_data must be [B, T, C]"
        B, T, C = in_data.shape
        t = np.arange(T, dtype=np.float32) / float(self.fs)
        import matplotlib.cm as cm
        colors = [cm.get_cmap("tab20", C)(i) for i in range(C)]

        imgs = []
        for b in range(B):
            sample = in_data[b].detach().cpu().numpy().astype(np.float32)  
            img = self._render_stacked_image(sample, t, colors)            
            img = self._pad_to_square(img, color=(255, 255, 255))
            x = self.image_preprocess(img).unsqueeze(0).to(self.device)    
            imgs.append(x)

        batch = torch.cat(imgs, dim=0)                 
        feats = self.model.encode_image(batch).float() 
        return feats

    # ---------- internals ----------
    def _render_stacked_image(self, sample_tc: np.ndarray, t: np.ndarray, colors) -> Image.Image:
        T, C = sample_tc.shape
        x_min, x_max = float(t[0]), float(t[-1])
        linewidth_pt = max(0.5, self.line_px) * 72.0 / float(self.dpi)

        panels = []
        for ch in range(C):
            fig = plt.figure(
                figsize=(self.panel_width / self.dpi, self.per_panel_height / self.dpi),
                dpi=self.dpi
            )
            ax = fig.add_subplot(111)
            ax.plot(t, sample_tc[:, ch], linewidth=linewidth_pt, color=colors[ch])
            ax.set_xlim(x_min, x_max)

            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)

            plt.tight_layout(pad=0.1)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.dpi)
            plt.close(fig)
            buf.seek(0)
            panels.append(Image.open(buf).convert("RGB"))

        target_w = max(im.size[0] for im in panels)
        padded_panels = []
        for im in panels:
            if im.size[0] < target_w: 
                pad_w = target_w - im.size[0]
                im = ImageOps.expand(im, border=(0, 0, pad_w, 0), fill=(255, 255, 255))
            padded_panels.append(im)

        total_h = sum(im.size[1] for im in padded_panels)
        stacked = Image.new("RGB", (target_w, total_h), (255, 255, 255))
        y = 0
        for im in padded_panels:
            stacked.paste(im, (0, y))
            y += im.size[1]
        return stacked

    @staticmethod
    def _pad_to_square(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), color)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    @staticmethod
    def _infer_embed_dim(model) -> int:
        if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
            return int(model.visual.output_dim)
        if hasattr(model, "embed_dim"):
            return int(model.embed_dim)
        raise RuntimeError("Cannot infer CLIP embedding dimension.")
