from Katna.image_filters.text_detector import TextDetector
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import sys

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

def ocr():
    image_documents = SimpleDirectoryReader("./data/images").load_data()

    OPENAI_API_TOKEN="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"
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=1500
    )

    response_1 = openai_mm_llm.complete(
        prompt="Perform OCR on the pages shown. Only provide text from the images in the response. Give the response as json, with the text under a key called output",
        # prompt="Perform OCR on the pages extract text using and return the text in a json format with the page number as the key.",
        image_documents=image_documents,
    )
    return response_1

def main():
    video_file_path = os.path.join(sys.argv[1])

    # assume like 0.75 flips per second
    vid_info = helper.get_video_info(video_file_path)
    num_frames = int(vid_info[1] * 0.75)
    extract(video_file_path, num_frames, "output_dir")


if __name__ == "__main__":
    main()
