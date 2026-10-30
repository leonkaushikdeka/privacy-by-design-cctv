"""Builds the case-study report (docs/Privacy_CCTV_Report.docx) from the notebook's results and figures."""
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
SHOTS = DOCS / "notebook_screens"
R = json.loads((DOCS / "results.json").read_text())
EV, LAT = R["eval"], R["latency"]
N_LABELLED = len(json.loads((ROOT / "data" / "eval" / "labels.json").read_text())["images"])
N_TESTS = sum(p.read_text().count("\ndef test_") for p in (ROOT / "tests").glob("test_*.py"))

AUTHOR = "Leon Kaushik Deka"
NAVY = RGBColor(0x1D, 0x24, 0x33)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
GREY = RGBColor(0x5B, 0x64, 0x75)

doc = Document()
fig_no = [0]


def setup_styles() -> None:
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        h = doc.styles[name]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.color.rgb = NAVY
        h.font.bold = True
        h.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        h.paragraph_format.space_after = Pt(4)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = s.bottom_margin = Cm(2.0)


def page_number_footer() -> None:
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def para(text: str, bold_lead: str | None = None, size: float | None = None, color=None, align=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align is not None:
        p.alignment = align
    return p


def bullets(items: list[str]) -> None:
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if it.startswith("**") and "**" in it[2:]:
            lead, rest = it[2:].split("**", 1)
            p.add_run(lead).bold = True
            p.add_run(rest)
        else:
            p.add_run(it)


def numbered(items: list[str]) -> None:
    for it in items:
        doc.add_paragraph(it, style="List Number")


def figure(path: Path, caption: str, width_cm: float = 16.0) -> None:
    if not path.exists():
        para(f"[missing figure: {path.name}]", color=RGBColor(0xB5, 0x41, 0x41))
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_no[0] += 1
    para(f"Figure {fig_no[0]}. {caption}", size=9, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)


def table(header: list[str], rows: list[list], widths: list[float] | None = None) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, h in zip(t.rows[0].cells, header):
        c.text = ""
        c.paragraphs[0].add_run(str(h)).bold = True
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    doc.add_paragraph()


def shot(cell: int) -> Path:
    return SHOTS / f"cell_{cell:02d}_code.png"


def title_page() -> None:
    for _ in range(4):
        doc.add_paragraph()
    para("MINI PROJECT / CASE STUDY REPORT", size=11, color=TEAL, align=WD_ALIGN_PARAGRAPH.CENTER).runs[0].bold = True
    para("Privacy-by-Design AI CCTV", size=28, color=NAVY, align=WD_ALIGN_PARAGRAPH.CENTER).runs[0].bold = True
    para("On-camera masking of intimate body regions and UPI PIN entry using pose estimation, "
         "body-part detection and hand tracking", size=13, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    hero = FIG / "anatomy-exploded-light.png"
    if hero.exists():
        doc.add_picture(str(hero), width=Cm(13))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    para("AI application area: AI-based image recognition (edge computer vision)", size=11,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para(f"Submitted by: {AUTHOR}", size=12, align=WD_ALIGN_PARAGRAPH.CENTER).runs[0].bold = True
    for line in ("Third Semester Mini Project", "Artificial Intelligence and Machine Learning (AIML)", "October 2026"):
        para(line, size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.paragraphs[-1].runs[0].add_break(WD_BREAK.PAGE)


def contents() -> None:
    doc.add_heading("Contents", level=1)
    for i, s in enumerate(["Introduction", "Problem Statement", "AI Approach", "Data", "Techniques and Algorithms",
                           "Working Process", "Implementation and Demonstration", "Results and Applications",
                           "Advantages and Limitations", "Ethical Considerations", "Future Scope", "Conclusion"], 1):
        para(f"{i}.  {s}")
    para("References")
    para("Appendix A: Jupyter notebook screenshots")
    para("Appendix B: Data sources and licences")
    doc.paragraphs[-1].runs[0].add_break(WD_BREAK.PAGE)


def abstract() -> None:
    doc.add_heading("Abstract", level=1)
    para("There are CCTV cameras everywhere: hostels, banks, shops. The issue is that these cameras often "
         "record too much. Take a CCTV camera in a hostel corridor. It records residents in personal "
         "clothing. Or a camera above a billing counter. It records customers entering their UPI PIN. And "
         "who can access this video? Anyone with access to the control room, or an attacker who can hack "
         "into the system.")
    para("This is a project to create an AI/ML CCTV pipeline that only masks the sensitive areas in the "
         "video. All the frames go through the AI chip in the camera. This means that there is no way to "
         "store or transmit video without the masking. We use pose estimation along with a pre-trained "
         "body part detector to locate and mask nipples, buttocks and genitalia. We have a second pipeline"
         " that detects phones and POS machines and with the help of hand tracking, we can detect when a "
         "PIN is being entered and mask both the screen and the hand. Masks are permanent. Since there is "
         "no forensic value in a PIN or a private body part, the rest of the video is still good as "
         "evidence.")
    para(f"A laptop-and-webcam prototype was built and tested on {N_LABELLED} hand-labelled, "
         f"openly licensed images. With the default policy it fully protected {EV['Set A: sensitive regions protected (>=90% masked)']} "
         f"PIN-entry regions, raised {EV['Set B: false alarms (device, no PIN entry)']} false alarms on images of idle "
         f"devices and {EV['Set C: false alarms (clothed people)']} on ordinary clothed people, and processed "
         f"640x480 frames of a busy scene at {LAT['fps']} fps on a laptop CPU with no GPU or NPU. Every miss "
         "traced back to a hand or a "
         "device that the pre-trained detectors did not find, which sets out clear next steps.")


def s1_intro() -> None:
    doc.add_heading("1. Introduction", level=1)
    para("Video surveillance is one of the most common camera uses we come across in daily life. Hostels, "
         "banks, ATMs, retail billing counters, hospitals and offices all record continuously, and the "
         "recordings are kept for weeks. Today's IP cameras are mini computers: they have a system-on-chip"
         " (SoC) with an image signal processor (ISP), a video encoder and, more and more often, a neural "
         "processing unit (NPU) designed to run deep-learning models on the camera itself (edge AI).")
    para("Usually on-camera AI is used to add surveillance features like people counting or intrusion "
         "alerts. This project turns the same hardware around and uses it to remove data the camera should"
         " never have captured. It's an application of AI-based image recognition: four pre-trained "
         "computer vision models (human pose estimation, body-part detection, object detection and hand "
         "landmark tracking) combined with rule-based fusion, temporal reasoning and irreversible image "
         "masking.")
    para("The main principle is privacy by design [15]. Privacy is baked into the design, not added as a "
         "policy after the fact. And if you mask before encoding, inside the camera, no unmasked recording"
         " exists anywhere. Nothing to leak, hand over under subpoena, misuse or steal.")


def s2_problem() -> None:
    doc.add_heading("2. Problem Statement", level=1)
    para("CCTV cameras record more than they need to, and the extra is the most sensitive part:")
    bullets([
        "**Hostels and dormitories.** Corridor cams. They see residents moving between rooms and "
        "bathrooms in towels or underwear. The camera is there for safety, not to watch bodies.",
        "**Billing counters, banks and ATMs.** A camera installed above the counter can capture the exact"
        " finger positions of a customer typing their UPI PIN on a phone, or their card PIN on a POS "
        "terminal. Once someone has the PIN, a stolen phone or card can be drained.",
        "**Access is wide.** It can be accessed by maintenance vendors, control room personnel and "
        "anybody else who gets an export. Also IP cameras are often a target of attack. The Mirai botnet "
        "exploited many IP cameras with default logins [12].",
    ])
    para("If you blur entire frames or turn the cameras off, you lose the security benefit the cameras "
         "exist for. So the challenge is to remove only the sensitive parts, in real time, in a way that "
         "cannot be reversed, while keeping the rest of the frame as evidence. Three measurable questions "
         "are used to judge the prototype:")
    numbered(["How well does it mask the sensitive areas (protection)?",
              "How often does it mask areas that did not need masking (over-masking / false alarms)?",
              "How quickly can it process frames in real time (latency / frames per second)?"])


def s3_approach() -> None:
    doc.add_heading("3. AI Approach", level=1)
    para("It's a per-frame vision pipeline meant to run on the camera's NPU, between the image sensor and "
         "the video encoder. There are two detection branches and a single masking stage shared by both.")
    bullets([
        "**Body privacy.** A pose estimation model finds the skeleton of each person. The skeleton tells "
        "us where a chest and a pelvis can be, and it guides a pre-trained body-part detector. The "
        "detector runs on each person at full resolution. Detections inside the matching anatomical "
        "region are accepted more easily, and anatomically impossible ones are discarded. Exposed "
        "nipples, buttocks and genitalia are pixelated.",
        "**PIN guard.** An object detector finds phones and POS terminals. A hand tracker gives 21 "
        "keypoints per hand. When a fingertip lands on the face of a device, the device and the hand are "
        "filled solid until the interaction ends.",
    ])
    para("Four design decisions turn detection into privacy protection:")
    bullets([
        "**Mask at the source.** Masks are written into the frame buffer before encoding, so the only "
        "video that exists is the masked video.",
        "**Permanent masks.** Solid fill or coarse pixelation, never Gaussian blur (which can be partly "
        "reversed).",
        "**Mask early, release late.** A region is masked as soon as there is evidence, and released only"
        " after 12 consecutive frames without evidence. This way detector flicker can't reveal individual"
        " frames.",
        "**Fail closed.** If any model fails on a frame, the whole frame is withheld instead of passed "
        "through.",
    ])
    para("Why do you need AI? Because the sensitive areas can move. And change shape. And show up anywhere"
         " in the frame. A static privacy zone (those black rectangles you set up in your camera settings)"
         " is not going to follow you as you walk down a hallway. And isn't going to follow the phone in "
         "your customer's hand. And it's obscuring way more of the picture than you need.")


def s4_data() -> None:
    doc.add_heading("4. Data", level=1)
    doc.add_heading("4.1 Data the system consumes", level=2)
    bullets([
        "**Input:** colour video frames from the camera sensor (the prototype uses 640x480 webcam "
        "frames).",
        "**Intermediate:** 33 keypoints for each person. 21 keypoints for each hand. Bounding boxes (with"
        " class, confidence score) for body parts, devices.",
        "**Output:** the same frames with the sensitive regions permanently masked, plus metadata only "
        "(number of masked regions, timings). No unmasked pixels are stored.",
    ])
    doc.add_heading("4.2 Data the models were trained on", level=2)
    para("All four models were pre-trained by their authors; no training was done in this project. "
         "EfficientDet-Lite0 was trained on COCO 2017 [5], which has 80 classes, including 'cell phone' "
         "and 'remote'. The MediaPipe pose and hand models were trained by Google on large annotated "
         "datasets of body and hand images [2, 3]. The NudeNet detector was trained by its author on a "
         "body-part dataset with 18 classes (exposed and covered) [7].")
    doc.add_heading("4.3 Test data used in this project", level=2)
    para("We did not attempt to gather any video of real people taking off their clothes or real PINs. We "
         "could not have done so without infringing on the privacy we want to protect. 83 openly licensed "
         "photos were downloaded from Wikimedia Commons (list of licences per image in Appendix B). 55 "
         "were manually labelled to form three sets:")
    table(["Set", "Images", "Meaning", "Label"],
          [["A", 8, "a hand operating a phone or POS terminal", "box over device face + operating fingers"],
           ["B", 26, "phone / POS / PIN pad visible, nobody entering anything", "must not be masked"],
           ["C", 21, "ordinary clothed people in corridors and shops", "must not be masked"]],
          [1.2, 1.6, 7.5, 5.5])
    para("Ambiguous regions, such as a customer scanning a QR code, are labelled 'ignore': masking them "
         "counts neither as protection nor as over-masking. A production system would also need consented,"
         " annotated CCTV recordings of real counters and corridors, plus a labelled POS-terminal dataset."
         " The webcam app in this project can record such data with frame-level ground truth (see Section "
         "7).")


def s5_techniques() -> None:
    doc.add_heading("5. Techniques and Algorithms", level=1)
    doc.add_heading("5.1 Pre-trained models", level=2)
    table(["Task", "Model", "Architecture", "Input", "Output", "Size"],
          [["Pose estimation", "MediaPipe Pose Landmarker (lite)", "BlazePose: person detector + landmark "
            "regression CNN [2]", "256x256", "33 keypoints / person", "5.8 MB"],
           ["Body parts", "NudeNet v3 '320n' [7]", "YOLOv8n single-stage detector, ONNX", "320x320",
            "18 classes (exposed / covered)", "12.2 MB"],
           ["Phones / POS", "EfficientDet-Lite0, int8 [4]", "EfficientNet-Lite backbone + BiFPN", "320x320",
            "COCO boxes: cell phone, remote", "4.6 MB"],
           ["Hand tracking", "MediaPipe Hand Landmarker [3]", "BlazePalm detector + landmark CNN", "192/224 px",
            "21 keypoints / hand", "7.8 MB"]],
          [2.4, 3.3, 4.0, 1.6, 3.0, 1.4])
    doc.add_heading("5.2 Algorithms developed in this project", level=2)
    bullets([
        "**Pose-guided ROI cropping.** Each person's chest-to-thigh box is cropped and passed to the "
        "body-part detector, so a person at the end of a long corridor is seen at the detector's full 320"
        " px instead of a few pixels.",
        "**Region-aware thresholding.** The pipeline uses shoulder and hip keypoints to work out chest "
        "and pelvis areas. A breast detection inside a chest area, or a genitalia/buttocks detection "
        "inside a pelvis area, is accepted at a score of 0.25. Anywhere else it needs 0.45.",
        "**Vetoes against detector errors.** A body-part detection that lies mostly on a tracked hand is "
        "rejected unless it sits inside a person's matching body region (the detector mistakes fingers "
        "for genitalia). On an upright person, a pelvis part above the shoulders or a chest part above "
        "the nose is anatomically impossible and is rejected. Skeletons with implausible proportions "
        "aren't trusted (the pose model sometimes builds a 'person' out of two hands).",
        "**Pose-anchored persistence.** Each accepted body part is stored relative to the person's hip "
        "centre and torso length. So if the detector misses a frame, the mask moves with the body.",
        "**PIN-entry rule.** A hand is entering a PIN when an index, middle or thumb fingertip touches "
        "the device face (device box + 12 %). An extended pointing index finger, a thumb on a held phone,"
        " or fingertip tapping over the last 10 frames raises the confidence. The rule is deliberately "
        "high-recall.",
        "**Guided zoom passes.** Small hands and phones get a second look at full model resolution: "
        "device -> crop for the hand, pose wrist -> crop for the hand, hand -> crop for the device. Only "
        "targets smaller than 20 % of the frame height are zoomed, with at most 3 crops per pass on still"
        " images and 1 per frame on live video (cycling through candidates; the temporal hold covers the "
        "gap).",
        "**Parallel branches.** Pose -> body parts, the device detector and the hand tracker don't depend"
        " on each other within a frame, so they run in parallel, like separate engines on a camera SoC.",
        "**Call filter.** A device sitting on a head (a pose head box or a confidently detected face) is "
        "a phone call, not PIN entry. Grazing the chin is not enough: the device centre must be on the "
        "head, or most of the device must overlap it.",
        "**Occlusion memory.** The last device box is stored relative to the wrist. While the hand hides "
        "the phone, the device mask moves with the hand.",
        "**Temporal hold.** Mask regions are tracked by IoU matching and kept for 12 frames after the "
        "last detection.",
        "**Irreversible masking and fail-closed.** PIN regions are filled solid, because even a pixelated"
        " finger can show which row of keys is being pressed. Body regions are pixelated to about 6 cells"
        " across. Any exception blanks the whole frame.",
    ])
    doc.add_heading("5.3 Per-frame algorithm", level=2)
    code = ("for each frame F from the sensor:\n"
            "    try:\n"
            "        people  = pose(F);         parts   = body_parts(F) + body_parts(crop(F, roi(p)) for p in people)\n"
            "        devices = detect(F);       hands   = hand_tracker(F)\n"
            "        hands  += zoom_hands(F, devices, wrists(people));  devices += zoom_devices(F, hands)\n"
            "        body = gate(parts, regions(people), veto_hands=hands) + anchored_parts(people)\n"
            "        pin  = [hand U device for hand, device if fingertip_on(device, hand)\n"
            "                and not at_head(device)] + carried_devices(hands)\n"
            "        masks = hold(body) + hold(pin)          # 12-frame release delay\n"
            "        F[masks] = solid / pixelate              # in place, before encoding\n"
            "    except error:\n"
            "        F[:] = 0                                 # fail closed\n"
            "    encoder.write(F)")
    p = doc.add_paragraph()
    r = p.add_run(code)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)


def s6_process() -> None:
    doc.add_heading("6. Working Process", level=1)
    para("The figure below follows one frame from input to output. Both branches work on every frame. The "
         "PIN guard gets pose keypoints (wrist crops, head boxes), and the body branch gets the tracked "
         "hands (hand veto).")
    figure(FIG / "pipeline_flow.png", "Per-frame workflow inside the camera's NPU boundary.", 16.5)
    doc.add_heading("6.1 Inside the camera", level=2)
    para("In the exploded view below you can see where the pipeline sits in an actual AI camera: light "
         "comes through the front window and lens, the IR-cut filter switches for day and night, and the "
         "CMOS image sensor sends raw frames through a flexible cable to the main SoC. The ISP turns them "
         "into colour frames, the NPU runs the privacy pipeline and writes masks into the frame buffer, "
         "and only after that does the H.265 encoder read the frame for the microSD card and the PoE "
         "network port. There is no physical path from the sensor to storage or the network that bypasses "
         "the NPU stage.")
    figure(FIG / "anatomy-exploded-light.png", "Exploded view of the privacy-by-design AI camera: front window, IR "
           "LED ring, M12 lens, image sensor board, main board (SoC + NPU), I/O board (PoE, microSD), rear cap and "
           "cover. Interactive version: anatomy/cctv-camera/cctv-camera.html.", 15)
    figure(FIG / "anatomy-closeup-light.png", "Close-up of the boards. The frame tile has left the sensor board on "
           "the flat flex cable and sits on the NPU with its sensitive regions already blacked out; only this "
           "masked frame continues to the encoder and the I/O board.", 13)
    doc.add_heading("6.2 Step by step", level=2)
    numbered([
        "Capture: the sensor delivers a frame into a buffer that only the pipeline can read.",
        "Pose: up to 4 skeletons are estimated; chest, pelvis, head and wrist positions are derived.",
        "Body parts: the detector runs on the whole frame and on each person crop; detections pass the gates and vetoes.",
        "Devices and hands: phones/POS terminals and hands are detected, then re-checked with zoom crops.",
        "PIN rule: fingertip-on-device interactions become PIN masks (calls excluded, hidden devices carried).",
        "Hold: new regions mask immediately; old regions persist 12 frames after their last evidence.",
        "Mask: regions are filled or pixelated in place. On any error the frame is blanked instead.",
        "Encode: the masked frame is compressed and written to storage or streamed. The raw buffer is overwritten by the next frame.",
    ])


def s7_impl() -> None:
    doc.add_heading("7. Implementation and Demonstration", level=1)
    para("Python 3.12 with OpenCV, MediaPipe Tasks and ONNX Runtime. Runs on a laptop CPU, with a webcam "
         "standing in for the camera sensor. All models run locally. No cloud.")
    table(["Path", "Contents"],
          [["privacy_cctv/detectors/", "wrappers for the four pre-trained models (vectorised NudeNet decoder)"],
           ["privacy_cctv/body_privacy.py", "pose-guided cropping, gating, vetoes, anchored persistence"],
           ["privacy_cctv/pin_guard.py", "PIN-entry rule, call filter, occlusion memory, tapping cue"],
           ["privacy_cctv/memory.py, masking.py", "temporal hold; irreversible masking"],
           ["privacy_cctv/pipeline.py", "per-frame orchestration, zoom passes, fail-closed, stage timings"],
           ["privacy_cctv/edge_camera.py", "simulated camera boundary: only masked frames leave"],
           ["privacy_cctv/metrics.py", "protection, over-masking, false alarms, leak rate, latency"],
           ["app/run_camera.py", "live webcam prototype with recording and ground-truth logging"],
           ["notebooks/privacy_cctv_demo.ipynb", "stage-by-stage demonstration and evaluation"],
           ["tests/", f"{N_TESTS} unit tests (geometry, masking, hold, PIN rule, call filter, fail-closed)"],
           ["anatomy/cctv-camera/", "interactive exploded view of the camera"]],
          [5.5, 11.0])
    doc.add_heading("7.1 How to run", level=2)
    run = ("python -m venv .venv  &&  .venv\\Scripts\\pip install -r requirements.txt\n"
           "python scripts/fetch_assets.py --samples        # models + sample images\n"
           "jupyter lab notebooks/privacy_cctv_demo.ipynb   # demonstration and evaluation\n"
           "python app/run_camera.py --source 0 --profile retail --log data/runs/demo.csv\n"
           f"python -m pytest tests                          # {N_TESTS} tests")
    p = doc.add_paragraph()
    r = p.add_run(run)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    para("In the live app, press g while you are actually typing a PIN to mark ground truth. The per-frame"
         " CSV log then gives frame-level protection, false masking and FPS for your own recording. The "
         "last notebook cell analyses it.")
    doc.add_heading("7.2 Demonstration (Jupyter notebook)", level=2)
    para("The notebook executes each step on real images. Here are some chosen results; all cells can be "
         "found in Appendix A.")
    figure(shot(5), "Notebook: pose keypoints and the chest and pelvis regions derived from them.", 12)
    figure(shot(9), "Notebook: pose-gated body masking, demonstrated with the covered classes used as "
                    "stand-ins (no explicit imagery was used anywhere in this project).", 16)
    figure(shot(11), "Notebook: PIN guard. Left of each pair: hands and devices found; right: the only "
                     "output that leaves the camera.", 16)
    figure(shot(13), "Notebook: the device detector is forced to miss after frame 0, and the device mask is "
                     "carried with the hand.", 15)
    figure(shot(15), "Notebook: masking styles. Blur keeps thousands of colours and is partly reversible; the "
                     "pipeline's pixelation and solid fill keep 30 and 1.", 16)
    figure(shot(17), "Notebook: fail-closed. A simulated NPU fault withholds the whole frame.", 13)


def s8_results() -> None:
    doc.add_heading("8. Results and Applications", level=1)
    doc.add_heading("8.1 Protection and over-masking", level=2)
    table(["Metric", "Result"], [[k, v] for k, v in EV.items()], [11.5, 4.0])
    figure(FIG / "eval_bars.png", "Left: share of each Set A sensitive region masked (dashed line: 90 % "
                                  "protection bar). Right: false-alarm rate on Sets B and C.", 16)
    figure(FIG / "eval_gallery.png", "Camera output for Set A images (coverage in each title).", 16)
    para("Looking at the misses: all Set A failures are detection failures, not fusion errors. In one "
         "image neither the hand nor the phone was found (a driver photographed through a windscreen). In "
         "another the phone was found, but the hand model missed a hand coming out of a sleeve cuff. And "
         "in the most CCTV-like image (customers at a shop counter, phone about 20 pixels across) the hand"
         " and phone were too small even for the zoom passes. Whenever both the hand and the device were "
         "detected, the region was protected, or very nearly. The only remaining Set B false alarm is a "
         "vendor holding a QR-code stand, which the COCO detector reads as a phone.")
    doc.add_heading("8.2 Policies and profiles", level=2)
    if R.get("policies"):
        pol = R["policies"]
        table(list(pol[0].keys()), [list(r.values()) for r in pol])
        by = {r["policy"]: r for r in pol}
        nz = by.get("default without zoom passes")
        if nz:
            para(f"Ablation: switching the guided zoom passes off lowers Set A mean coverage from "
                 f"{by['interaction-gated (default)']['Set A mean coverage']} to {nz['Set A mean coverage']}, "
                 f"while Set B false alarms go from {by['interaction-gated (default)']['Set B false alarms']} to "
                 f"{nz['Set B false alarms']}. The zoom passes find small hands and phones the full-frame models "
                 "miss, and the one false alarm they add is the QR-code stand.")
        para("If you mask every device (the bank profile), you catch the PIN entries whose hand was never "
             "detected, but you also mask idle devices. That's fine for an ATM or PIN-pad zone.")
    if R.get("profiles"):
        prof = R["profiles"]
        table(list(prof[0].keys()), [list(r.values()) for r in prof])
    doc.add_heading("8.3 Temporal hold", level=2)
    figure(FIG / "hold_curve.png", "Share of PIN-entry frames left unmasked against the hold length, for "
                                   "three per-frame detection rates and two miss patterns (simulation).", 16)
    table(list(R["hold"][0].keys()), [list(r.values()) for r in R["hold"]])
    para("Without a hold, the leak rate is the same as the detector's miss rate. With a 12 frame hold "
         "(0.8s at 15fps), random misses are pretty much completely filtered out. Bursty misses, where the"
         " hand hides the device for a few frames, still leak a bit, but that's exactly what the occlusion"
         " memory is aimed at.")
    doc.add_heading("8.4 Speed", level=2)
    st = LAT["stages"]
    table(["Stage", "Mean ms / frame"], [[k, v] for k, v in st.items()] +
          [["total (mean / p95)", f"{LAT['mean_ms']} / {LAT['p95_ms']}"], ["frames per second", LAT["fps"]]],
          [8, 5])
    figure(FIG / "latency.png", "Per-stage cost and per-frame latency on a simulated 640x480 camera pan "
                                "(laptop CPU, video mode).", 16)
    abl = LAT.get("ablation", [])
    if abl:
        table(["Configuration", "Mean ms", "p95 ms", "fps"],
              [[a["configuration"], a["mean_ms"], a["p95_ms"], a["fps"]] for a in abl], [7.5, 2.5, 2.5, 2])
        para(f"Running the three independent branches (pose -> body parts, device detector, hand tracker) in "
             f"parallel threads cuts the mean latency from {abl[0]['mean_ms']} ms to {abl[1]['mean_ms']} ms. "
             f"The zoom passes add about {round(abl[1]['mean_ms'] - abl[2]['mean_ms'])} ms on this busy scene "
             "(several small hands), and less on quiet scenes.")
    para(f"On the laptop CPU the full pipeline runs at {LAT['fps']} fps on a busy scene. That is usable for a demonstration, "
         "but below the 15 fps of typical CCTV recording. The four networks dominate the cost. They total about "
         "30 MB and are standard architectures (BlazePose, YOLOv8n, EfficientDet-Lite, BlazePalm) that "
         "commonly run in real time on edge accelerators after int8 quantisation, so meeting real-time "
         "speed on camera hardware is a deployment step, not a change of method.")
    doc.add_heading("8.5 Applications", level=2)
    bullets(["**Hostels, PGs and dormitories:** corridor cameras that keep safety value without recording "
             "bodies.",
             "**Retail billing counters and UPI payments:** masking customers' PIN entry on phones and POS "
             "terminals.",
             "**Banks and ATMs:** strict profile masking PIN pads continuously.",
             "**Hospitals and elder care:** patient dignity in wards and corridors.",
             "**Gyms, pools and changing-room approaches:** areas where partial undress is expected.",
             "**Any camera that exports footage:** police requests, insurance claims and social-media leaks "
             "carry no sensitive regions."])


def s9_adv() -> None:
    doc.add_heading("9. Advantages and Limitations", level=1)
    doc.add_heading("Advantages", level=2)
    bullets(["**Privacy by construction:** unmasked footage never exists outside the camera, so it cannot leak"
             " or be hacked.",
             "**Evidence preserved:** only regions without forensic value are removed; faces, actions and the "
             "scene remain.",
             "**Targeted:** sets B and C show near-zero masking of idle devices and clothed people.",
             "**Fail-safe:** model errors withhold frames; flicker is absorbed by the hold.",
             "**Configurable:** hostel, retail and bank profiles choose what counts as sensitive.",
             "**Low cost:** pre-trained models, CPU-only prototype, no cloud processing or extra bandwidth."])
    doc.add_heading("Limitations", level=2)
    bullets(["**Small evaluation set:** 55 labelled still images (8 positives). It shows behaviour, not a "
             "statistically strong accuracy estimate.",
             "**Detector recall bounds protection:** if the pre-trained models miss a hand or a phone, nothing"
             " is masked (3 of 8 positives). Small, distant subjects are the weak case.",
             "**Body pipeline recall not measured on nudity:** for ethical reasons no explicit images were "
             "used; only the false-alarm side was measured, on clothed people.",
             "**No dedicated POS class:** COCO has no POS-terminal class. The 'cell phone' class catches many "
             "terminals, but not all.",
             f"**Speed:** {LAT['fps']} fps on a laptop CPU, below real-time CCTV rates without an accelerator.",
             "**Pose model:** the lite model has trouble seeing far-away people and sometimes builds 'people' "
             "out of hands, so plausibility checks were added.",
             "**Untested conditions:** night-time IR footage, crowds, fisheye lenses and deliberate evasion."])


def s10_ethics() -> None:
    doc.add_heading("10. Ethical Considerations", level=1)
    bullets([
        "**Privacy and law.** In line with purpose-limitation and data-minimisation principles under "
        "India's Digital Personal Data Protection Act, 2023 [8] and the recognition of privacy as a "
        "fundamental right (Puttaswamy, 2017 [9]), the risk of unauthorised image capture of a private "
        "part of a person (offence under Section 66E of IT Act [10] and voyeurism offence under Section "
        "77 of Bharatiya Nyaya Sanhita [11]) can be mitigated for camera owners by source masking.",
        "**Fairness and bias.** Pose, hand and body-part models can perform differently across skin "
        "tones, body types, clothing styles and disabilities. A miss is a privacy failure for that "
        "person, so the system needs per-group testing on consented, diverse data before deployment. "
        "Masking is offered for all genders (male chest masking is configurable per site).",
        "**Security.** The masking stage has to be part of signed firmware, and the debug and raw-preview"
        " modes have to be removed from production builds, or an attacker could disable it. The "
        "prototype's model-view overlays exist only in development.",
        "**No function creep.** The pipeline only detects body parts and devices in order to remove them."
        " No identification, no face recognition, no tracking of individuals, and its output shouldn't be"
        " reused for that.",
        "**Transparency and consent.** Signage should say that the camera masks sensitive areas. A count "
        "of masked regions can be kept in the metadata, so auditors can check that the system was running"
        " without seeing the actual content.",
        "**Evidence integrity.** Masks are always permanent. The answer is no if an investigator wants a "
        "masked area. This trade-off should be documented for the legal team at the site.",
        "**Responsible data handling in this project.** No nudity and no real PINs were collected. Sample"
        " images are openly licensed and credited. Images that turned out to contain nudity were deleted.",
    ])


def s11_future() -> None:
    doc.add_heading("11. Future Scope", level=1)
    bullets(["**Dedicated POS / PIN-pad detector** fine-tuned (e.g. with MediaPipe Model Maker or YOLO) on "
             "consented counter recordings.",
             "**Multi-person pose** (YOLO-pose or RTMPose) for crowded corridors and small, distant people.",
             "**Edge deployment:** int8 quantisation and compilation for camera NPUs (e.g. Rockchip RV1126, "
             "Ambarella CVflow, Hailo-8L), with the masking stage inside the ISP-to-encoder path.",
             "**Temporal models:** classify PIN-typing from landmark sequences (TCN/LSTM) instead of single-"
             "frame rules.",
             "**Segmentation masks** instead of boxes, to hide even less of the scene.",
             "**Benchmark:** a consented, diverse, frame-labelled video dataset of counters and corridors, "
             "with per-group fairness metrics.",
             "**Tamper detection:** alert when the camera is covered, re-aimed or its firmware altered."])


def s12_conclusion() -> None:
    doc.add_heading("12. Conclusion", level=1)
    para("This project demonstrates that the existing AI hardware in modern CCTV cameras can be used to "
         "increase privacy instead of decreasing it. By utilizing pose estimation and a pre-trained body "
         "part detector, object detector, and hand tracking, the prototype hides private body parts and "
         "PIN entry in the camera itself (before any video is recorded or transferred) for good, yet "
         "leaves the rest of the scene available for evidence.")
    para(f"On labelled test images it protected {EV['Set A: sensitive regions protected (>=90% masked)']} PIN-entry "
         f"regions, with {EV['Set B: false alarms (device, no PIN entry)']} false alarms on idle devices and "
         f"{EV['Set C: false alarms (clothed people)']} on clothed people. A simulation showed that a 12-frame hold "
         "cuts the share of leaked PIN frames from 10-40 % (the detector's miss rate) to under 2 %, even when "
         "misses come in bursts. Every miss traced back to "
         "a hand or device the pre-trained detectors did not find, so the clear next steps are better small-object "
         "detection, a dedicated POS model and deployment on a camera NPU for real-time speed. The architecture "
         "(mask at the source, mask early and release late, fail closed) holds regardless of which detectors "
         "are plugged in.")


def references() -> None:
    doc.add_heading("References", level=1)
    refs = [
        "C. Lugaresi et al., \"MediaPipe: A Framework for Building Perception Pipelines,\" arXiv:1906.08172, 2019.",
        "V. Bazarevsky et al., \"BlazePose: On-device Real-time Body Pose Tracking,\" arXiv:2006.10204, 2020.",
        "F. Zhang et al., \"MediaPipe Hands: On-device Real-time Hand Tracking,\" arXiv:2006.10214, 2020.",
        "M. Tan, R. Pang, Q. V. Le, \"EfficientDet: Scalable and Efficient Object Detection,\" CVPR, 2020.",
        "T.-Y. Lin et al., \"Microsoft COCO: Common Objects in Context,\" ECCV, 2014.",
        "G. Jocher et al., Ultralytics YOLOv8, 2023. https://github.com/ultralytics/ultralytics",
        "P. Bedapudi, NudeNet: lightweight nudity detection (v3). https://github.com/notAI-tech/NudeNet",
        "Government of India, The Digital Personal Data Protection Act, 2023.",
        "Justice K. S. Puttaswamy (Retd.) v. Union of India, (2017) 10 SCC 1.",
        "Government of India, The Information Technology Act, 2000, Section 66E.",
        "Government of India, The Bharatiya Nyaya Sanhita, 2023, Section 77.",
        "M. Antonakakis et al., \"Understanding the Mirai Botnet,\" USENIX Security Symposium, 2017.",
        "Wikimedia Commons: test images and licences listed in data/samples/sources.json. https://commons.wikimedia.org",
        "Google, MediaPipe Tasks model cards (Pose Landmarker, Hand Landmarker, Object Detector). https://ai.google.dev/edge/mediapipe",
        "A. Cavoukian, \"Privacy by Design: The 7 Foundational Principles,\" Information and Privacy Commissioner of Ontario, 2009.",
        "ONNX Runtime (https://onnxruntime.ai) and OpenCV (https://opencv.org).",
    ]
    for i, ref in enumerate(refs, 1):
        p = para(f"[{i}] {ref}", size=9.5)
        p.paragraph_format.space_after = Pt(2)


def appendix_a() -> None:
    doc.add_page_break()
    doc.add_heading("Appendix A: Jupyter notebook screenshots", level=1)
    para("Screenshots of the executed notebook (notebooks/privacy_cctv_demo.ipynb), in order. Cells were "
         "run top to bottom on the same laptop used for the timing results.")
    for path in sorted(SHOTS.glob("cell_*_code.png")):
        idx = int(path.stem.split("_")[1])
        md = SHOTS / f"cell_{idx - 1:02d}_md.png"
        if md.exists():
            doc.add_picture(str(md), width=Cm(15.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_picture(str(path), width=Cm(15.5))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        para(f"Notebook cell {idx}", size=8.5, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)


def appendix_b() -> None:
    doc.add_page_break()
    doc.add_heading("Appendix B: Data sources and licences", level=1)
    src = json.loads((ROOT / "data" / "samples" / "sources.json").read_text())
    para(f"{len(src)} images from Wikimedia Commons. Licence counts below; per-image author, licence and page URL "
         "are in data/samples/sources.json.")
    counts = {}
    for v in src.values():
        counts[v.get("license") or "see page"] = counts.get(v.get("license") or "see page", 0) + 1
    table(["Licence", "Images"], sorted(counts.items(), key=lambda kv: -kv[1]), [8, 3])
    para("MediaPipe Pose Landmarker, Hand Landmarker and EfficientDet-Lite0 are released by Google (with "
         "model cards); and NudeNet 320n is released in NudeNet repository. Please consult each upstream "
         "licence for commercial purposes. The anatomy figure kit (wheresryan22/anatomy) is MIT licensed.")


def main() -> None:
    doc.core_properties.author = AUTHOR
    doc.core_properties.last_modified_by = AUTHOR
    doc.core_properties.title = "Privacy-by-Design AI CCTV"
    setup_styles()
    page_number_footer()
    title_page()
    contents()
    abstract()
    s1_intro()
    s2_problem()
    s3_approach()
    s4_data()
    s5_techniques()
    s6_process()
    s7_impl()
    s8_results()
    s9_adv()
    s10_ethics()
    s11_future()
    s12_conclusion()
    references()
    appendix_a()
    appendix_b()
    out = DOCS / f"Privacy_CCTV_Report_{AUTHOR.replace(' ', '_')}.docx"
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
