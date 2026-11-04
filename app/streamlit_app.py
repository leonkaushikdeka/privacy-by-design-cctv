"""Web front end for the privacy pipeline.

    .venv\\Scripts\\streamlit run app/streamlit_app.py

Live camera: the browser sends webcam frames to this server over WebRTC, the pipeline masks them,
and only the masked frames are sent back to the page. Nothing is stored.
"""
from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "3")

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import av  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_webrtc import WebRtcMode, webrtc_streamer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privacy_cctv import PROFILES, PrivacyPipeline  # noqa: E402
from privacy_cctv.edge_camera import REGION_STYLE, hud, outline, state_of  # noqa: E402
from privacy_cctv.viz import overlay  # noqa: E402

AUTHOR = "Leon Kaushik Deka"
PROFILE_HELP = {
    "demo": "every phone, POS terminal and held card is masked; PIN entry filled solid",
    "retail": "PIN entry on phones and POS terminals, plus body privacy",
    "bank": "every payment device and card, all the time",
    "hostel": "body privacy only, including underwear-level regions",
    "all": "default settings: PIN entry and body privacy",
}
RTC = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
SAMPLES = ["ua_00.jpg", "phone_00.jpg", "phone_01.jpg", "pos_00.jpg", "ue_00.jpg", "people_04.jpg"]
MAX_VIDEO_SECONDS = 20


def make_config(profile: str, body: bool, video: bool):
    cfg = PROFILES[profile].with_(video_mode=video, zoom_crops=1 if video else 3)
    return cfg if body else cfg.with_(body_privacy=False)


def fit_width(img: np.ndarray, width: int = 640) -> np.ndarray:
    if img.shape[1] == width:
        return img
    h = round(img.shape[0] * width / img.shape[1]) // 2 * 2
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


class LiveMasker:
    """Called by streamlit-webrtc on its own thread for every webcam frame."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.key = None
        self.pipeline: PrivacyPipeline | None = None
        self.labels, self.mirror = True, True
        self.stats = {"fps": 0.0, "ms": 0.0, "state": "camera off", "masks": 0}
        self._t0 = time.monotonic()
        self._last = time.perf_counter()

    def configure(self, profile: str, body: bool, labels: bool, mirror: bool) -> None:
        old = None
        with self.lock:
            self.labels, self.mirror = labels, mirror
            if self.key != (profile, body):
                old, self.pipeline = self.pipeline, PrivacyPipeline(make_config(profile, body, video=True))
                self.key = (profile, body)
        if old is not None:
            old.close()

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = fit_width(frame.to_ndarray(format="bgr24"))
        if self.mirror:
            img = cv2.flip(img, 1)
        with self.lock:
            if self.pipeline is None:
                return av.VideoFrame.from_ndarray(np.zeros_like(img), format="bgr24")
            r = self.pipeline.process(img, int((time.monotonic() - self._t0) * 1000))
            if self.labels:
                outline(r)
        now = time.perf_counter()
        fps = 0.9 * self.stats["fps"] + 0.1 / max(now - self._last, 1e-6)
        self._last = now
        self.stats = {"fps": fps, "ms": r.timings_ms.get("total", 0.0), "state": state_of(r), "masks": len(r.regions)}
        return av.VideoFrame.from_ndarray(hud(r, fps), format="bgr24")


@st.cache_resource(max_entries=2, show_spinner="Loading models...")
def image_pipeline(profile: str, body: bool) -> PrivacyPipeline:
    return PrivacyPipeline(make_config(profile, body, video=False))


@st.fragment(run_every=1.0)
def live_stats(masker: LiveMasker) -> None:
    s = masker.stats
    a, b = st.columns(2)
    a.metric("Frames per second", f"{s['fps']:.1f}")
    b.metric("Latency per frame", f"{s['ms']:.0f} ms")
    st.metric("Status", s["state"])


def region_table(regions) -> list[dict]:
    return [{"masked as": REGION_STYLE.get(g.source, ("PRIVATE",))[0], "detector label": g.label,
             "x1": round(g.x1), "y1": round(g.y1), "x2": round(g.x2), "y2": round(g.y2)} for g in regions]


def sidebar() -> tuple[str, bool, bool, bool]:
    with st.sidebar:
        st.markdown(f"**Privacy-by-Design AI CCTV**  \n{AUTHOR}  \nThird Semester AIML Mini Project")
        profile = st.selectbox("Profile", list(PROFILE_HELP), help="Which regions count as sensitive.")
        st.caption(PROFILE_HELP[profile])
        labels = st.toggle("Label masked regions", value=True)
        body = st.toggle("Body privacy", value=True, help="Turn off for a faster phone and card demo.")
        mirror = st.toggle("Mirror the webcam", value=True)
        st.divider()
        st.caption("Only masked frames reach this page. Masks are written into the frame before it is sent, "
                   "and any model error blanks the whole frame.")
    return profile, body, labels, mirror


LEGEND = [
    ("PHONE / POS", "pixelated", "a phone or payment terminal in view"),
    ("CARD", "pixelated", "a bank card held in a hand"),
    ("PIN ENTRY", "solid", "a fingertip on the phone or keypad"),
    ("PRIVATE", "pixelated", "exposed intimate body region"),
]


def live_tab(profile: str, body: bool, labels: bool, mirror: bool) -> None:
    masker: LiveMasker = st.session_state.setdefault("masker", LiveMasker())
    masker.configure(profile, body, labels, mirror)
    video, side = st.columns([3, 2], gap="large")
    with video:
        webrtc_streamer(
            key="live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=masker,
            rtc_configuration=RTC,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            video_html_attrs={"style": {"width": "100%"}, "autoPlay": True, "controls": False, "muted": True},
            async_processing=True,
        )
    with side:
        st.write("Press **Start** and allow the camera. Then hold up a phone, a POS terminal or a bank card, "
                 "and try typing on the phone.")
        live_stats(masker)
        st.dataframe([{"label": a, "mask": b, "when": c} for a, b, c in LEGEND], hide_index=True)


def mask_image(img: np.ndarray, profile: str, body: bool, labels: bool) -> None:
    pipe = image_pipeline(profile, body)
    pipe.reset()
    raw = fit_width(img)
    r = pipe.process(raw.copy())
    if labels:
        outline(r)
    left, right = st.columns([3, 2], gap="large")
    left.image(r.frame, channels="BGR", caption=f"Camera output: {state_of(r)}", width="stretch")
    with right:
        st.metric("Masked regions", len(r.regions))
        st.metric("Processing time", f"{r.timings_ms.get('total', 0):.0f} ms")
        if r.regions:
            st.dataframe(region_table(r.regions), hide_index=True, width="stretch")
        with st.expander("Model view (development only, the camera never outputs this)"):
            st.image(overlay(raw, r), channels="BGR", width="stretch")


def mask_video(data: bytes, suffix: str, profile: str, body: bool, labels: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / f"in{suffix}", Path(tmp) / "out.mp4"
        src.write_bytes(data)
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), int(fps * MAX_VIDEO_SECONDS)) or 1
        pipe = PrivacyPipeline(make_config(profile, body, video=True))
        out = av.open(str(dst), mode="w")
        stream = None
        bar = st.progress(0.0, text="Masking video...")
        try:
            for i in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                r = pipe.process(fit_width(frame), int(i * 1000 / fps))
                if labels:
                    outline(r)
                if stream is None:
                    stream = out.add_stream("libx264", rate=round(fps))
                    stream.width, stream.height, stream.pix_fmt = r.frame.shape[1], r.frame.shape[0], "yuv420p"
                for packet in stream.encode(av.VideoFrame.from_ndarray(r.frame, format="bgr24")):
                    out.mux(packet)
                bar.progress((i + 1) / total, text=f"Masking video... frame {i + 1} of {total}")
            if stream is not None:
                for packet in stream.encode():
                    out.mux(packet)
        finally:
            out.close()
            cap.release()
            pipe.close()
        bar.empty()
        left, right = st.columns([3, 2], gap="large")
        left.video(dst.read_bytes())
        right.metric("Frames processed", f"{i + 1}")
        right.download_button("Download masked video", dst.read_bytes(), "masked.mp4", "video/mp4")


def file_tab(profile: str, body: bool, labels: bool) -> None:
    up = st.file_uploader("Upload an image or a short video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "webm"])
    sample = st.selectbox("...or pick a sample image", ["(none)"] + SAMPLES)
    if up is not None:
        data, suffix = up.getvalue(), Path(up.name).suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png"):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                st.error("That file could not be read as an image.")
                return
            mask_image(img, profile, body, labels)
        else:
            st.caption(f"Only the first {MAX_VIDEO_SECONDS} seconds are processed.")
            mask_video(data, suffix, profile, body, labels)
    elif sample != "(none)":
        mask_image(cv2.imread(str(ROOT / "data" / "samples" / sample)), profile, body, labels)


def results_tab() -> None:
    results = json.loads((ROOT / "docs" / "results.json").read_text())
    ev, lat = results["eval"], results["latency"]
    cols = st.columns(4)
    for col, (label, key) in zip(cols, [
        ("PIN entries fully masked", "Set A: sensitive regions protected (>=90% masked)"),
        ("Mean coverage", "Set A: mean coverage of sensitive regions"),
        ("False alarms: idle devices", "Set B: false alarms (device, no PIN entry)"),
        ("False alarms: clothed people", "Set C: false alarms (clothed people)"),
    ]):
        col.metric(label, ev[key])
    st.caption(f"55 hand-labelled, openly licensed images. Speed on a busy 640x480 scene, laptop CPU: "
               f"{lat['mean_ms']} ms per frame ({lat['fps']} fps).")
    figs = ROOT / "docs" / "figures"
    left, right = st.columns(2)
    left.image(str(figs / "eval_bars.png"), caption="Coverage per PIN-entry image, and false-alarm rates")
    right.image(str(figs / "hold_curve.png"), caption="Frames left unmasked against the temporal hold length")
    left.image(str(figs / "latency.png"), caption="Where the time goes")
    right.image(str(figs / "anatomy-exploded-light.png"), caption="Inside the camera")
    report = next((ROOT / "docs").glob("Privacy_CCTV_Report_*.pdf"), None)
    if report:
        st.download_button("Download the full report (PDF)", report.read_bytes(), report.name, "application/pdf")


def main() -> None:
    st.set_page_config(page_title="Privacy-by-Design AI CCTV", page_icon=":material/videocam:", layout="wide")
    profile, body, labels, mirror = sidebar()
    st.title("Privacy-by-Design AI CCTV")
    st.caption("On-camera masking of intimate body regions, phone and POS screens, payment cards and PIN entry.")
    live, files, results = st.tabs(["Live camera", "Try a file", "Results"])
    with live:
        live_tab(profile, body, labels, mirror)
    with files:
        file_tab(profile, body, labels)
    with results:
        results_tab()


main()
