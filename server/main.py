import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.formatters import TextFormatter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_ID_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
)


class TranscriptRequest(BaseModel):
    url: str


def extract_video_id(url: str) -> str | None:
    match = VIDEO_ID_PATTERN.search(url)
    return match.group(1) if match else None


@app.post("/api/transcript")
def get_transcript(request: TranscriptRequest):
    video_id = extract_video_id(request.url)
    if not video_id:
        return {"error": "Could not find a YouTube video ID in that URL."}

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(["en"])
        data = transcript.fetch()
        raw = list(data)  # materialise before iterating twice
        text = TextFormatter().format_transcript(raw)
        def _seg(s):
            if isinstance(s, dict):
                return {"text": s.get("text", ""), "start": s.get("start", 0)}
            return {"text": getattr(s, "text", ""), "start": getattr(s, "start", 0)}
        segments = [_seg(s) for s in raw]
        return {"videoId": video_id, "transcript": text, "segments": segments}
    except TranscriptsDisabled:
        return {"error": "Transcripts are disabled for this video."}
    except NoTranscriptFound:
        return {"error": "No English transcript is available for this video."}
    except VideoUnavailable:
        return {"error": "This video is unavailable."}
    except Exception as exc:
        return {"error": f"Failed to fetch transcript: {exc}"}
