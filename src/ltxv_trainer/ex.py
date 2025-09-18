# from ltxv_trainer.ltxv_utils import decode_video
# import torch
# from ltxv_trainer.model_loader import load_ltxv_components

# # 1. VAE 로드
# components = load_ltxv_components(
#     model_source="Lightricks/LTX-Video-0.9.5",
#     load_text_encoder_in_8bit=False,
#     transformer_dtype=torch.bfloat16,
#     vae_dtype=torch.bfloat16,
# )
# vae = components.vae.to("cuda").eval()

# # # 2. 랜덤 latent 생성
# shape = (2, 2016, 128)  # [B, seq_len, D]
# latents = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
# latents = latents[0:1]

# # 3. Decode
# decoded = decode_video(
#     vae=vae,
#     latents=latents,
#     num_frames=6,
#     height=14,
#     width=24,
#     device="cuda",
#     dtype=torch.bfloat16,
# )
# print("Decoded shape:", decoded.shape)  # 보통 (3, F, H, W)
# print("VAE latent_channels:", vae.config.latent_channels)


import torch 

shape = (1,10,3)

t = torch.randn(shape)
a,b = t.chunk(2, dim = 1)
print(a.shape)