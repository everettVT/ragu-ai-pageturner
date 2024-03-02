from Katna.image_filters.text_detector import TextDetector
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import sys

import Katna.helper_functions as helper


# td = TextDetector()
# td.download()


def extract(video_file_path: str, num_frames: int, output_path: str):
    vd = Video()
    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=output_path)
    print(f"Input video file path = {video_file_path}")
    vd.extract_video_keyframes(
        no_of_frames=num_frames, file_path=video_file_path,
        writer=diskwriter
    )


def main():
    video_file_path = os.path.join(sys.argv[1])

    # assume like 0.75 flips per second
    vid_info = helper.get_video_info(video_file_path)
    num_frames = int(vid_info[1] * 0.75)
    extract(video_file_path, num_frames, "output_dir")


if __name__ == "__main__":
    main()
