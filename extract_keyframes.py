from Katna.image_filters.text_detector import TextDetector
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import re
import sys
import typing
import time 

import Katna.helper_functions as helper

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from pymilvus import connections

import yaml

# td = TextDetector()
# td.download()


with open("config.yaml") as f:
    config = yaml.safe_load(f)

openai_key = config['openai']['key']


def extract(video_file_path: str, num_frames: int, output_path: str):
    vd = Video()
    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=output_path)
    print(f"Input video file path = {video_file_path}")
    vd.extract_video_keyframes(
        no_of_frames=num_frames, file_path=video_file_path,
        writer=diskwriter
    )

def ingest(text: str):
    pass


def ocr(img_paths: typing.List[str]):
    image_documents = SimpleDirectoryReader(img_paths).load_data()
    responses = []

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=openai_key, max_new_tokens=1500
    )
    for image_doc in image_documents:
        try:
            response_1 = openai_mm_llm.complete(
                prompt="Perform OCR on the pages shown. Only provide text from the images in the response. Give the response as json, with the text under a key called output",
                image_documents=[image_doc],
            )
            # print(response_1)
            json_txt = response_1.text.replace('```', '').replace("\'", '')
            json_txt = json_txt.replace('\\n', ' ').replace('\\\\n', ' ')
            json_txt = re.findall(r'.*output["\']\s*:\s*([^}]+)', json_txt)
            print(json_txt)
            if len(json_txt[0].strip()) > 5:
                responses.append(json_txt[0])
        except Exception as e:
            print(e)
    return responses


def main():
    video_file_path = os.path.join(sys.argv[1])

    # assume like 0.70 flips per second
    vid_info = helper.get_video_info(video_file_path)
    num_frames = int(vid_info[1] * 0.70)
    output_dir = sys.argv[2]
    extract(video_file_path, num_frames, output_dir)
    ocr_op = ocr(output_dir)
    print(ocr_op)

    txt_dir = f"{output_dir}/text/"
    os.makedirs(txt_dir, exist_ok=True)
    for i, s in enumerate(ocr_op):
        with open(os.path.join(txt_dir, f"{i}.txt"), mode='wt') as f:
            f.write(s)
    print(txt_dir)


if __name__ == "__main__":
    main()
