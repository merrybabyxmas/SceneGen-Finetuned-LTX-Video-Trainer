#!/usr/bin/env python3

"""
Fix for trainer.py validation decoding to match clean decode_precomputed.py output.

The issue is that trainer.py uses direct vae.decode() call which may add noise,
while decode_precomputed.py uses ltxv_utils.decode_video() which properly handles
decode_timestep=0.0 and decode_noise_scale=0.0 for clean output.
"""

# Here's the fix to apply to trainer.py decode_and_save_video function:

"""
REPLACE THIS CODE IN TRAINER.PY (around line 894-896):

OLD CODE:
    with autocast(debug_device.type, dtype=torch.bfloat16):
        timestep = torch.zeros(1, device=debug_device, dtype=torch.long)
        result = self._vae.decode(reshaped / self._vae.config.scaling_factor, timestep, return_dict=False)

NEW CODE:
    with autocast(debug_device.type, dtype=torch.bfloat16):
        # Use decode_video function for clean output (same as decode_precomputed.py)
        from ltxv_trainer.ltxv_utils import decode_video
        
        # Convert back to [B, L, D] format for decode_video
        latents_for_decode = reshaped.permute(0, 2, 3, 4, 1).reshape(1, -1, vae_channels)
        
        result = decode_video(
            vae=self._vae,
            latents=latents_for_decode,
            num_frames=batch_F,
            height=batch_H, 
            width=batch_W,
            device=debug_device,
            patch_size=1,
            patch_size_t=1,
            decode_timestep=0.0,  # No noise
            decode_noise_scale=0.0,  # No noise
            generator=None
        )
        result = (result,)  # Wrap in tuple to match expected format
"""

print("""
찾은 문제:

1. decode_precomputed.py: ltxv_utils.decode_video() 사용 → 노이즈 없음
   - decode_timestep=0.0, decode_noise_scale=0.0

2. trainer.py: 직접 vae.decode() 호출 → 노이즈 추가됨  
   - VAE 내부에서 timestep에 따른 처리로 노이즈 발생

해결방법:
trainer.py의 decode_and_save_video 함수를 수정해서
ltxv_utils.decode_video() 함수를 사용하도록 변경하면 깨끗한 출력을 얻을 수 있습니다.

위의 코드 변경사항을 trainer.py 894-896줄 주변에 적용하세요.
""")