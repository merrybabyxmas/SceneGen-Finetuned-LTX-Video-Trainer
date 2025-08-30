

    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        print(f"batch : {batch}")
        # 1) Latents 언팩
        curr_lat, F, H, W, fps = _unpack_latent_entry(batch["latent_conditions"])

        prev_lat = None
        if batch.get("prev_conditions", None) is not None:
            prev_lat, _, _, _, _ = _unpack_latent_entry(batch["prev_conditions"])

        # 2) Conditions 언팩 (scene-level이면 curr와 동일)
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["text_conditions"])
        
        
        # print(f"-------------batch config-------------"
        #       f"curr_lat : {curr_lat.shape}"
        #       f"prev_lat : {prev_lat.shape}"
        #       f"prompt_embeds : {prompt_embeds.shape}"
        #       )

        # 3) 노이즈 샘플 & 시그마 (curr에만 적용)
        sigmas = timestep_sampler.sample_for(curr_lat)         # (B, Cseq, 1) 또는 전략 구현에 따라
        # ↓ 아래 연산에서 (B,1,1) 브로드캐스트를 기대하므로 reshape
        if sigmas.dim() > 3:
            raise ValueError("Unexpected sigma shape; expected (B,1,1) style broadcast.")
        sigmas = sigmas.view(curr_lat.shape[0], 1, 1)          # (B,1,1)

        noise = torch.randn_like(curr_lat, device=curr_lat.device)  # (B, Cseq, D)
        noisy_curr = (1 - sigmas) * curr_lat + sigmas * noise       # (B, Cseq, D)

        # 4) curr 내부 '첫 프레임 conditioning' 적용: 첫 프레임 토큰은 클린으로 대체
        first_mask_curr = self._create_first_frame_conditioning_mask(
            batch_size=curr_lat.shape[0],
            sequence_length=curr_lat.shape[1],
            height=H,
            width=W,
            device=curr_lat.device,
        )  # (B, Cseq) True=clean keep
        noisy_curr = torch.where(first_mask_curr.unsqueeze(-1), curr_lat, noisy_curr)

        # 5) prev + curr concat
        concat_lat, Pseq, Cseq = _concat_prev_curr(prev_lat, noisy_curr)  # (B, P+C, D)

        # 6) conditioning mask 구성
        #    prev 전체 True(조건), curr은 first_frame만 True
        if Pseq > 0:
            prev_mask = torch.ones(curr_lat.shape[0], Pseq, dtype=torch.bool, device=curr_lat.device)  # (B, Pseq)=True
            conditioning_mask = torch.cat([prev_mask, first_mask_curr], dim=1)  # (B, P+C)
        else:
            conditioning_mask = first_mask_curr  # (B, C)

        # 7) 타깃 구성: prev 구간은 0 (마스킹되므로 영향 X), curr 구간은 (noise - clean)
        targets_curr = noise - curr_lat  # (B, Cseq, D)
        if Pseq > 0:
            zeros_prev = torch.zeros(curr_lat.shape[0], Pseq, curr_lat.shape[2], device=curr_lat.device, dtype=targets_curr.dtype)
            targets = torch.cat([zeros_prev, targets_curr], dim=1)  # (B, P+C, D)
        else:
            targets = targets_curr  # (B, C, D)

        # 8) timestep 생성: prev=0, curr=round(sigmas*1000)
        sampled_t = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()  # (B,)
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_t)  # (B, P+C)

        # 9) ROPE scale & video coords (prev+curr 길이에 맞추어 준비)
        rope_scale = get_rope_scale_factors(fps)
        # seq_mult = 2 if Pseq > 0 else 1  # prev를 붙이면 2배
        
        
        if Pseq > 0:
            raw_coords = prepare_video_coordinates(
                num_frames=F, height=H, width=W,
                batch_size=concat_lat.shape[0],
                sequence_multiplier=2,              # ★ 핵심: prev + curr
                device=concat_lat.device,
            )
            prescaled_f = raw_coords[..., 0] * rope_scale[0]
            prescaled_h = raw_coords[..., 1] * rope_scale[1]
            prescaled_w = raw_coords[..., 2] * rope_scale[2]
            video_coords = torch.stack([prescaled_f, prescaled_h, prescaled_w], dim=1)  # (B, 3, P+C)
        else:
            video_coords = None

        # return TrainingBatch(
        #     latents=concat_lat,
        #     targets=targets,
        #     prompt_embeds=prompt_embeds,
        #     prompt_attention_mask=prompt_attention_mask,
        #     timesteps=timesteps,
        #     sigmas=sigmas,
        #     conditioning_mask=conditioning_mask,
        #     num_frames=F,
        #     height=H,
        #     width=W,
        #     fps=fps,
        #     rope_interpolation_scale=rope_scale,
        #     video_coords=video_coords,
        # )
        
        return TrainingBatch(
            latents=concat_lat,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=F,
            height=H,
            width=W,
            fps=fps,
            rope_interpolation_scale=rope_scale,
            video_coords=video_coords,
        )
        
        
        
        
        
        
        ### 샷 1개인 경우
    def prepare_batch(self, batch: dict[str, Any], timestep_sampler: TimestepSampler) -> TrainingBatch:
        # 1) Latents 언팩 (curr만)
        curr_lat, F, H, W, fps = _unpack_latent_entry(batch["latent_conditions"])

        # 2) Conditions 언팩
        prompt_embeds, prompt_attention_mask = _unpack_condition_entry(batch["text_conditions"])

        # 3) 노이즈 샘플 & 시그마
        sigmas = timestep_sampler.sample_for(curr_lat)
        sigmas = sigmas.view(curr_lat.shape[0], 1, 1)

        noise = torch.randn_like(curr_lat, device=curr_lat.device)
        noisy_curr = (1 - sigmas) * curr_lat + sigmas * noise

        # 4) curr 첫 프레임 conditioning
        first_mask_curr = self._create_first_frame_conditioning_mask(
            batch_size=curr_lat.shape[0],
            sequence_length=curr_lat.shape[1],
            height=H,
            width=W,
            device=curr_lat.device,
        )
        noisy_curr = torch.where(first_mask_curr.unsqueeze(-1), curr_lat, noisy_curr)

        # 5) concat 대신 curr만 사용
        concat_lat = noisy_curr
        Pseq, Cseq = 0, curr_lat.shape[1]

        # 6) conditioning mask = curr의 first_frame
        conditioning_mask = first_mask_curr  # (B, C)

        # 7) targets
        targets = noise - curr_lat  # (B, C, D)

        # 8) timesteps
        sampled_t = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_t)

        # 9) ROPE & coords (curr만 → coords None 처리)
        rope_scale = get_rope_scale_factors(fps)
        video_coords = None

        return TrainingBatch(
            latents=concat_lat,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=F,
            height=H,
            width=W,
            fps=fps,
            rope_interpolation_scale=rope_scale,
            video_coords=video_coords,
        )