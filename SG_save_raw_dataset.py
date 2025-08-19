# SG_save_raw_dataset.py 
import os
import json
import time
import random
import argparse
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

# =========================
# FFmpeg / FFprobe helpers
# =========================
def _probe_duration_seconds(video_path: Path) -> float:
    """ffprobe로 동영상 길이(초)를 읽어옴. 실패 시 0.0"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(video_path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def extract_middle_frame(video_path: Path, out_jpg: Path) -> bool:
    """
    동영상 절반 지점으로 시킹해서 프레임 1장 추출.
    - ffprobe로 duration 찾고
    - ffmpeg -ss <mid> -i <in> -frames:v 1 <out>
    """
    try:
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        dur = _probe_duration_seconds(video_path)
        mid = max(0.0, dur / 2.0)  # dur==0이면 0초

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{mid:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_jpg)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_jpg.exists()
    except Exception:
        return False


# =========================
# Caption helpers
# =========================
def build_caption(item: dict, query: str, vf: dict) -> str:
    """
    우선순위:
    1) Pexels alt(제목)이 있으면 그걸 사용 (불필요한 'by xxx' 꼬리표 제거)
    2) 없으면 검색어(query)를 사용 (예: "city night")
    + 보조정보: orientation, WxH, duration(s) 를 괄호로 보강
    """
    # 1) alt 정리
    title = (item.get("alt") or "").strip()
    if title:
        lower = title.lower()
        cut_tokens = [" by ", " | pexels", " - pexels"]
        for tok in cut_tokens:
            idx = lower.find(tok)
            if idx != -1:
                title = title[:idx].strip()
                break

    # 2) 보조정보
    w, h = vf.get("width"), vf.get("height")
    dur = item.get("duration")  # 초 단위 가능성 높음
    orient = None
    if isinstance(w, int) and isinstance(h, int):
        if w > h:
            orient = "landscape"
        elif w < h:
            orient = "portrait"
        else:
            orient = "square"

    meta_bits = []
    if orient:
        meta_bits.append(orient)
    if isinstance(w, int) and isinstance(h, int):
        meta_bits.append(f"{w}x{h}")
    if isinstance(dur, (int, float)) and dur > 0:
        meta_bits.append(f"{int(dur)}s")

    meta = ", ".join(meta_bits)
    meta_str = f" ({meta})" if meta else ""

    # 3) 최종 캡션
    if title:
        cap = f"{title}{meta_str}"
    else:
        cap = f"{query}{meta_str}"

    return cap


# =========================
# Pexels API & download
# =========================
def pick_video_file(video_files, target_height=720):
    """
    Pexels 'video_files' 중 학습에 적당한 mp4 하나를 고름.
    높이(target_height) 근처의 mp4를 우선, 없으면 가장 작은 파일(대략 가벼운 것)을 반환.
    """
    mp4s = [vf for vf in video_files if (vf.get("file_type", "") or "").lower() == "video/mp4"]
    if not mp4s:
        return None
    with_h = [vf for vf in mp4s if vf.get("height")]
    if with_h:
        with_h.sort(key=lambda x: abs(x["height"] - target_height))
        return with_h[0]
    mp4s.sort(key=lambda x: x.get("width", 10**9))
    return mp4s[0]


def download(url, out_path: Path, headers, max_retries=3, timeout=60):
    for attempt in range(max_retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            if attempt + 1 == max_retries:
                print(f"[!] Failed to download {url}: {e}")
                return False
            time.sleep(1.5 * (attempt + 1))


def pexels_search(api_key, query, per_page=40, page=1):
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": min(max(per_page, 1), 80),
        "page": max(page, 1),
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json(), headers


# =========================
# (옵션) BLIP2 캡셔닝
# =========================
_blip2_cache = {"processor": None, "model": None}

def load_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    if _blip2_cache["model"] is None:
        _blip2_cache["processor"] = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        _blip2_cache["model"] = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto"
        )
    return _blip2_cache["processor"], _blip2_cache["model"]


def blip2_caption(image_path: Path, prompt="Describe the video frame in one short sentence."):
    from PIL import Image
    import torch
    from transformers import Blip2Processor
    processor, model = load_blip2()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    txt = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return txt.strip()


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build LTXV dataset.json from Pexels")
    parser.add_argument("--api_key", type=str, default="PIXEL.COM_API_KEY", help="Pexels API key")
    parser.add_argument("--query", type=str, default="nature", help="Search keyword")
    parser.add_argument("--total", type=int, default=200, help="Total videos to fetch")
    parser.add_argument("--per_page", type=int, default=40, help="Pexels per_page (1~80)")
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--target_height", type=int, default=720)
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between pages")
    parser.add_argument("--use_blip2", action="store_true", help="중간 프레임에 대해 BLIP2 자동 캡션 생성")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    frames_dir = out_dir / "frames"
    dataset_json = out_dir / "dataset.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    collected = 0
    page = 1

    pbar = tqdm(total=args.total, desc="Collecting videos")
    while collected < args.total:
        data, headers = pexels_search(args.api_key, args.query, per_page=args.per_page, page=page)
        items = data.get("videos", [])
        if not items:
            print("[!] No more results from Pexels. Try another query or smaller total.")
            break

        random.shuffle(items)  # 다양화
        for item in items:
            if collected >= args.total:
                break

            vf = pick_video_file(item.get("video_files", []), target_height=args.target_height)
            if not vf:
                continue

            video_url = vf.get("link")
            if not video_url:
                continue

            # 파일명: pexels-{id}-{width}x{height}.mp4
            vid = item.get("id")
            w, h = vf.get("width"), vf.get("height")
            filename = f"pexels-{vid}-{w}x{h}.mp4"
            out_path = videos_dir / filename

            ok = download(video_url, out_path, headers={})  # 파일 URL은 공개이므로 별도 header 불필요
            if not ok:
                continue

            # 캡션 생성
            if args.use_blip2:
                jpg_path = frames_dir / (filename.replace(".mp4", ".jpg"))
                if extract_middle_frame(out_path, jpg_path):
                    try:
                        auto_cap = blip2_caption(jpg_path)
                        cap = auto_cap if auto_cap else build_caption(item, args.query, vf)
                    except Exception:
                        cap = build_caption(item, args.query, vf)
                else:
                    cap = build_caption(item, args.query, vf)
            else:
                cap = build_caption(item, args.query, vf)

            results.append({
                "caption": cap,
                "media_path": f"videos/{filename}",
            })
            collected += 1
            pbar.update(1)

        page += 1
        time.sleep(args.sleep)

    pbar.close()

    # 저장
    with open(dataset_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Saved {len(results)} items -> {dataset_json}")
    if results:
        print(f"   Example record: {results[0]}")
    print(f"   Videos dir: {videos_dir.resolve()}")
    if args.use_blip2:
        print(f"   Frames dir: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
