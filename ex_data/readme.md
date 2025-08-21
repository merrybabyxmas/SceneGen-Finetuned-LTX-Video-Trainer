[TODO 정선 : 데이터셋 구축하기]
1. raw videos 
2. metadata.json

<video 저장 형식>
data/
└─ videos/
   ├─ a001.mp4
   ├─ a002.mp4
   └─ b103.mp4

<metadata 형식>
[
  {"caption": "sentence1", "media_path": "data/videos/a001.mp4"},
  {"caption": "sentence2", "media_path": "data/videos/a002.mp4"}
]



[split videos : 동우]
1. splitted videos
2. split_metatdata.json



splits/
└─ a001/
   ├─ a001_shot0.mp4
   ├─ a001_shot1.mp4
   └─ a001_shot2.mp4
└─ a002/
   ├─ a002_shot0.mp4
   └─ a002_shot1.mp4

keyframes/              # (옵션, visualize_keyframes=True일 때)
└─ a001/
   ├─ EOS_35.jpg
   └─ SOS_36.jpg

splits/split_meta.json  # 샷 범위 & 파일 경로 기록



<split metadata.json 형식> ##shot path 가 "media path"로써 활용댐
[
  {
    "orig_media_path": "data/videos/a001.mp4",
    "caption": "문장1",                 // 원본 캡션을 샷마다 반복 사용
    "shot_index": 0,
    "start_frame": 0,
    "end_frame": 35,
    "shot_path": "splits/a001/a001_shot0.mp4",
    "fps": 24.0,
    "H": 1080,
    "W": 1920,
    "T": 312
  },
  {
    "orig_media_path": "data/videos/a001.mp4",
    "caption": "문장1",
    "shot_index": 1,
    "start_frame": 36,
    "end_frame": 120,
    "shot_path": "splits/a001/a001_shot1.mp4",
    "fps": 24.0,
    "H": 1080,
    "W": 1920,
    "T": 312
  }
]



[TODO 동우 : split_meta.json -> metadata_shots.json 으로 변환]

<metadata_shots.json 예시>
[
  {"caption": "문장1", "media_path": "splits/a001/a001_shot0.mp4", "shot_index": 0},
  {"caption": "문장1", "media_path": "splits/a001/a001_shot1.mp4", "shot_index": 1},
  {"caption": "문장2", "media_path": "splits/a002/a002_shot0.mp4", "shot_index": 0}
]







[precompute]

.precomputed/
├─ latents/
│  └─ .../a001_shot0.pt
│  └─ .../a001_shot1.pt
├─ conditions/
│  └─ .../a001_shot0.pt
│  └─ .../a001_shot1.pt
└─ decoded_videos/            # (옵션) 디코드 검증용

