from pathlib import Path
import random
import shutil
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "my_videos" / "incorrect_belt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_NAME = "incorrect_belt"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FRAME_RATE = 1  # сколько кадров в секунду извлекать

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def prepare_class_dirs():
    """
    Удаляет только старые папки incorrect_belt, если они уже были созданы.
    Остальные классы correct_belt и no_belt не трогает.
    """
    for split in ["train", "val", "test"]:
        class_dir = OUTPUT_DIR / split / CLASS_NAME

        if class_dir.exists():
            shutil.rmtree(class_dir)

        class_dir.mkdir(parents=True, exist_ok=True)


def get_videos():
    videos = [
        path for path in VIDEO_DIR.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    videos.sort()

    return videos


def split_videos(videos):
    """
    Делим именно видеофайлы, а не отдельные кадры.
    Это нужно, чтобы тестовая выборка была честной.
    """
    random.seed(42)
    random.shuffle(videos)

    total = len(videos)

    train_count = round(total * TRAIN_RATIO)
    val_count = round(total * VAL_RATIO)

    train_videos = videos[:train_count]
    val_videos = videos[train_count:train_count + val_count]
    test_videos = videos[train_count + val_count:]

    return {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }


def extract_frames(video_path: Path, split: str):
    target_dir = OUTPUT_DIR / split / CLASS_NAME

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps <= 0:
        print(f"Не удалось определить FPS: {video_path.name}")
        cap.release()
        return 0

    frame_interval = int(video_fps / FRAME_RATE)
    frame_interval = max(frame_interval, 1)

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_interval == 0:
            output_name = f"{video_path.stem}_{saved_count:05d}.jpg"
            output_path = target_dir / output_name

            cv2.imwrite(str(output_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()

    return saved_count


def main():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Папка с видео не найдена: {VIDEO_DIR}")

    videos = get_videos()

    if not videos:
        raise RuntimeError(f"В папке нет видео: {VIDEO_DIR}")

    print(f"Найдено видео: {len(videos)}")

    prepare_class_dirs()

    split_videos_dict = split_videos(videos)

    total_saved = 0

    for split, video_list in split_videos_dict.items():
        print()
        print(f"{split.upper()}: видеофайлов {len(video_list)}")

        split_saved = 0

        for video_path in video_list:
            saved_count = extract_frames(video_path, split)
            split_saved += saved_count
            print(f"{video_path.name}: сохранено кадров {saved_count}")

        total_saved += split_saved
        print(f"Итого кадров для {split}: {split_saved}")

    print()
    print(f"Всего сохранено кадров класса {CLASS_NAME}: {total_saved}")
    print("Добавление третьего класса завершено.")


if __name__ == "__main__":
    main()