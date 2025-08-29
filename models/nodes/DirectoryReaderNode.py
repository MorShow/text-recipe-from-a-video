from source.config import RAW_DATA
from models.elements import VideoElement

from pathlib import Path


def load_videos(input_dir: str | Path) -> dict[str, VideoElement]:
    base_path = Path(input_dir)
    videos_dict = dict()

    for video_file in base_path.rglob("*.mkv"):
        video_element = VideoElement(video_file)
        video_element.initiate()
        videos_dict[video_element.video_id] = video_element

    return videos_dict


if __name__ == "__main__":
    directory = RAW_DATA / 'youcookII' / 'training'
    videos = load_videos(directory)
    print(f"Loaded {len(videos)} videos:")
    for vid, ve in videos.items():
        print(f" - {vid}: {ve.video_seconds}s, {ve.frame_count} frames @ {ve.frame_ratio} fps")
