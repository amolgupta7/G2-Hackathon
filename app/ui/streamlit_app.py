"""C8: Streamlit search UI (thin client of the C7 API).

Run from the project root, with the API running:
    python -m streamlit run app/ui/streamlit_app.py
"""
import os

import requests
import streamlit as st

from app.config import AUDIO_DIR
from app.log import get_logger

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ALL = "All recordings"
ANY = "Any speaker"
log = get_logger("app.ui")


def mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_recordings() -> list[dict]:
    r = requests.get(f"{API_URL}/recordings", timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_search(q: str, recording: str | None, speaker: str | None, n: int) -> dict:
    params = {"q": q, "n": n}
    if recording:
        params["recording"] = recording
    if speaker:
        params["speaker"] = speaker
    r = requests.get(f"{API_URL}/search", params=params, timeout=30)
    if r.status_code == 422:
        detail = r.json().get("detail")
        raise ValueError(detail if isinstance(detail, str) else "; ".join(d["msg"] for d in detail))
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Transcript Search", layout="wide")
st.title("Transcript Search")
st.caption("Hybrid keyword + semantic search across two-speaker conversations. "
           'Keywords support "exact phrases" and -exclude; questions work too.')

try:
    recordings = fetch_recordings()
except requests.RequestException as e:
    log.error("API unreachable at %s: %s", API_URL, e)
    st.error(f"Search API is not reachable at {API_URL}. Start it with "
             "`python -m uvicorn app.api.main:app --port 8000`.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    rec_ids = [r["id"] for r in recordings]
    recording = st.selectbox("Recording", [ALL] + rec_ids)
    # Speaker labels A/B only identify a person within one recording, so the filter needs a recording.
    speaker = st.selectbox("Speaker", [ANY, "A", "B"], disabled=recording == ALL,
                           help="Pick a recording first: speaker labels are per recording.")
    n = st.slider("Results", min_value=1, max_value=20, value=10)
    st.divider()
    st.caption(f"{len(recordings)} recordings · {sum(r['chunks'] for r in recordings)} speaker turns indexed")

with st.form("search"):
    q = st.text_input("Search", placeholder='e.g. refund · "connection pool" · why was the customer billed twice')
    submitted = st.form_submit_button("Search", type="primary")

if submitted and q.strip():
    rec = None if recording == ALL else recording
    spk = None if speaker == ANY or rec is None else speaker
    try:
        data = fetch_search(q.strip(), rec, spk, n)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    except requests.RequestException as e:
        log.error("search request failed: %s", e)
        st.error("Search failed. Check that the API and database are running.")
        st.stop()

    log.info("UI search q=%r recording=%s speaker=%s -> %d hits", q, rec, spk, data["count"])
    st.write(f"**{data['count']}** results · {data['took_ms']:.0f} ms")
    if data["count"] == 0:
        st.info("No matches. Try fewer or different words, or clear the filters.")

    for i, hit in enumerate(data["results"], start=1):
        with st.container(border=True):
            st.markdown(f"**{i}. {hit['recording_id']}** · Speaker **{hit['speaker']}** · "
                        f"{mmss(hit['start_s'])}–{mmss(hit['end_s'])}")
            if hit["prev"]:
                st.caption(f"{hit['prev']['speaker']}: {hit['prev']['text']}")
            st.markdown(f"> **{hit['speaker']}:** {hit['text']}")
            if hit["next"]:
                st.caption(f"{hit['next']['speaker']}: {hit['next']['text']}")
            audio = AUDIO_DIR / f"{hit['recording_id']}.wav"
            if audio.exists():
                st.audio(str(audio), start_time=int(hit["start_s"]))
            else:
                st.caption("Audio file not found.")
