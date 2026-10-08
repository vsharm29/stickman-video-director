from agno.tools.function import Function

# Enable direct invocation on Agno Function tool objects (tool(*args, **kwargs))
if "__call__" not in Function.__dict__:
    Function.__call__ = lambda self, *args, **kwargs: self.entrypoint(*args, **kwargs)

from src.tools.flow_generator import generate_flow_clip
from src.tools.video_stitcher import stitch_clips
from src.tools.caption_engine import generate_srt_file, embed_captions_to_video

__all__ = [
    "generate_flow_clip",
    "stitch_clips",
    "generate_srt_file",
    "embed_captions_to_video",
]
