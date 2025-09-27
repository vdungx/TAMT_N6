import os
import subprocess
from multiprocessing import Pool, cpu_count

ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
ffprobe_path = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffprobe.exe"

video_dir = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\20bn-something-something-v2"
output_dir = r"C:\Users\dungs\Documents\Study\ManageBigData\TAMT-main\datasets\SSV2\frames"

os.makedirs(output_dir, exist_ok=True)

def process_video(file):
    if not file.endswith(".webm"):
        return
    video_path = os.path.join(video_dir, file)
    name = os.path.splitext(file)[0]
    out_path = os.path.join(output_dir, name)
    os.makedirs(out_path, exist_ok=True)

    # ---- đếm số frame bằng ffprobe ----
    probe_cmd = [
        ffprobe_path, "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        total_frames = int(result.stdout.strip())
    except:
        print(f"⚠️ Không đếm được frame cho {file}, bỏ qua.")
        return

    # ---- tính 5 vị trí đều nhau ----
    if total_frames < 5:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / 5
        frame_indices = [int(i * step) for i in range(5)]

    expr = "+".join([f"eq(n\\,{idx})" for idx in frame_indices])

    cmd = [
        ffmpeg_path, "-i", video_path,
        "-vf", f"select='{expr}'", "-vsync", "vfr",
        os.path.join(out_path, "frame_%02d.jpg")
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    files = os.listdir(video_dir)
    n_workers = max(1, cpu_count() - 1)  # dùng (số core - 1) để tránh treo máy
    print(f"⚡ Dùng {n_workers} tiến trình song song")

    with Pool(n_workers) as p:
        p.map(process_video, files)
