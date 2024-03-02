from Katna.image_filters.text_detector import TextDetector
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import re
import sys
import typing

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

def ingest(text: str):
    pass


def ocr(img_paths: typing.List[str]):
    image_documents = SimpleDirectoryReader(img_paths).load_data()
    responses = []

    for image_doc in image_documents:
        OPENAI_API_TOKEN="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"
        openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
        )

        response_1 = openai_mm_llm.complete(
            prompt="Perform OCR on the pages shown. Only provide text from the images in the response. Give the response as json, with the text under a key called output",
            image_documents=[image_doc],
        )
        json_txt = response_1.text.replace('```', '').replace("\'", '')
        print(re.findall(r'output"\s+:\s+(.+)', json_txt))
        responses.append(json_txt)
    return responses


def main():
    video_file_path = os.path.join(sys.argv[1])

    # assume like 0.75 flips per second
    vid_info = helper.get_video_info(video_file_path)
    num_frames = 2  #int(vid_info[1] * 0.75)
    output_dir = "output_dir2"
    #extract(video_file_path, num_frames, output_dir)
    ocr_op = ocr(output_dir)
    print(ocr_op)


if __name__ == "__main__":
    main()
