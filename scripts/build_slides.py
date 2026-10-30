"""Builds the 12-slide presentation (docs/Privacy_CCTV_Slides.pptx) from the notebook's results and figures."""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
SHOTS = DOCS / "notebook_screens"
R = json.loads((DOCS / "results.json").read_text())
EV, LAT = R["eval"], R["latency"]

AUTHOR = "Leon Kaushik Deka"
NAVY = RGBColor(0x1D, 0x24, 0x33)
INK = RGBColor(0x1D, 0x24, 0x33)
MUTED = RGBColor(0x5B, 0x64, 0x75)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)
ROSE = RGBColor(0xD3, 0x5D, 0x8A)
AMBER = RGBColor(0xD9, 0x9A, 0x1F)
PAPER = RGBColor(0xF6, 0xF4, 0xEF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def bg(slide, color) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def rect(slide, x, y, w, h, color, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    if line:
        s.line.color.rgb = line
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    lines = runs if isinstance(runs, list) else [runs]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.color.rgb, r.font.bold, r.font.name = Pt(size), color, bold, FONT
    return tb


def bullets(slide, x, y, w, h, items, size=15, color=INK, gap=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        lead, _, rest = it.partition("|")
        r = p.add_run()
        r.text = ("▪  " + lead) if rest else ("▪  " + it)
        r.font.size, r.font.color.rgb, r.font.name, r.font.bold = Pt(size), color, FONT, bool(rest)
        if rest:
            r2 = p.add_run()
            r2.text = " " + rest
            r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size), color, FONT
    return tb


def picture(slide, path: Path, x, y, max_w, max_h, border=False):
    if not path.exists():
        text(slide, x, y, max_w, Inches(0.4), f"[missing {path.name}]", 12, ROSE)
        return None
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    pic = slide.shapes.add_picture(str(path), x + (max_w - w) // 2, y + (max_h - h) // 2, w, h)
    if border:
        pic.line.color.rgb = RGBColor(0xD8, 0xDC, 0xE3)
        pic.line.width = Pt(0.75)
    return pic


def header(slide, n: int, kicker: str, title: str) -> None:
    bg(slide, WHITE)
    rect(slide, 0, 0, Inches(0.16), H, TEAL)
    text(slide, Inches(0.6), Inches(0.38), Inches(9), Inches(0.35), kicker.upper(), 12, TEAL, True)
    text(slide, Inches(0.6), Inches(0.68), Inches(12.2), Inches(0.8), title, 30, NAVY, True)
    text(slide, Inches(12.2), Inches(7.0), Inches(0.8), Inches(0.3), f"{n:02d}", 11, MUTED, align=PP_ALIGN.RIGHT)
    text(slide, Inches(0.6), Inches(7.0), Inches(8), Inches(0.3), f"Privacy-by-Design AI CCTV  ·  {AUTHOR}", 11, MUTED)


def card(slide, x, y, w, h, title, body, accent):
    rect(slide, x, y, w, h, PAPER)
    rect(slide, x, y, w, Inches(0.07), accent)
    text(slide, x + Inches(0.25), y + Inches(0.25), w - Inches(0.5), Inches(0.45), title, 17, NAVY, True)
    bullets(slide, x + Inches(0.25), y + Inches(0.8), w - Inches(0.5), h - Inches(1.0), body, 13, INK, 4)


def stat(slide, x, y, w, value, label, accent):
    rect(slide, x, y, w, Inches(1.35), PAPER)
    rect(slide, x, y, Inches(0.07), Inches(1.35), accent)
    text(slide, x + Inches(0.25), y + Inches(0.12), w - Inches(0.35), Inches(0.7), value, 30, NAVY, True)
    text(slide, x + Inches(0.25), y + Inches(0.8), w - Inches(0.35), Inches(0.5), label, 11, MUTED)


def shot(cell: int, part: str = "code") -> Path:
    return SHOTS / f"cell_{cell:02d}_{part}.png"


def s01_title():
    s = prs.slides.add_slide(BLANK)
    bg(s, NAVY)
    picture(s, FIG / "anatomy-exploded-dark.png", Inches(6.3), Inches(0.5), Inches(6.8), Inches(6.5))
    text(s, Inches(0.7), Inches(1.6), Inches(6), Inches(0.4), "MINI PROJECT  ·  AI-BASED IMAGE RECOGNITION",
         12, TEAL, True)
    text(s, Inches(0.7), Inches(2.1), Inches(6), Inches(1.9), ["Privacy-by-Design", "AI CCTV"], 46, WHITE, True)
    text(s, Inches(0.7), Inches(4.0), Inches(5.6), Inches(1.2),
         "On-camera masking of intimate body regions and UPI PIN entry with pose estimation, "
         "body-part detection and hand tracking", 16, RGBColor(0xC9, 0xD1, 0xDE))
    text(s, Inches(0.7), Inches(5.6), Inches(5.6), Inches(0.45), AUTHOR, 18, WHITE, True)
    text(s, Inches(0.7), Inches(6.05), Inches(5.6), Inches(0.8),
         ["Third Semester Mini Project  ·  AIML", "October 2026"], 13, RGBColor(0x9A, 0xA4, 0xB5))


def s02_problem():
    s = prs.slides.add_slide(BLANK)
    header(s, 2, "Problem statement", "CCTV records more than its purpose needs")
    card(s, Inches(0.6), Inches(1.75), Inches(4.0), Inches(2.9), "Hostel corridor",
         ["Residents walk to bathrooms in towels or underwear", "Footage exists for safety, not bodies",
          "Recordings kept for weeks"], ROSE)
    card(s, Inches(4.75), Inches(1.75), Inches(4.0), Inches(2.9), "Billing counter / ATM",
         ["Camera sees fingers typing the UPI or card PIN", "PIN + stolen phone = emptied account",
          "Shoulder-surfing, at scale"], TEAL)
    card(s, Inches(8.9), Inches(1.75), Inches(3.85), Inches(2.9), "Who can watch?",
         ["Control-room staff and vendors", "Anyone given an export", "Attackers: IP cameras are a top botnet target (Mirai)"],
         AMBER)
    text(s, Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.6),
         ["Goal: remove only the sensitive regions, in real time, irreversibly, inside the camera, "
          "and keep the rest of the frame as evidence.",
          "Judged on: (1) how well it masks sensitive areas  (2) how often it masks unnecessary areas  (3) speed."],
         15, NAVY)


def s03_approach():
    s = prs.slides.add_slide(BLANK)
    header(s, 3, "AI approach", "Two AI branches, one permanent mask, inside the camera")
    card(s, Inches(0.6), Inches(1.75), Inches(6.0), Inches(2.55), "Body privacy",
         ["Pose estimation finds each skeleton|-> chest and pelvis regions",
          "Pre-trained body-part detector|runs per person at full resolution",
          "Nipples, buttocks, genitalia|pixelated"], ROSE)
    card(s, Inches(6.75), Inches(1.75), Inches(6.0), Inches(2.55), "PIN guard",
         ["Object detector|finds phones and POS terminals",
          "Hand tracking (21 points)|fingertip on device face = PIN entry",
          "Screen + hand|filled solid"], TEAL)
    principles = [("Mask at the source", "no unmasked recording exists"),
                  ("Permanent masks", "solid / pixelate, never blur"),
                  ("Mask early, release late", "12-frame hold beats flicker"),
                  ("Fail closed", "model error = frame withheld")]
    for i, (a, b) in enumerate(principles):
        x = Inches(0.6) + i * Inches(3.06)
        rect(s, x, Inches(4.6), Inches(2.9), Inches(1.6), NAVY)
        text(s, x + Inches(0.2), Inches(4.8), Inches(2.5), Inches(0.5), a, 16, WHITE, True)
        text(s, x + Inches(0.2), Inches(5.35), Inches(2.5), Inches(0.8), b, 13, RGBColor(0xC9, 0xD1, 0xDE))


def s04_anatomy():
    s = prs.slides.add_slide(BLANK)
    header(s, 4, "Inside the camera", "Every frame passes the NPU before it can be stored or sent")
    picture(s, FIG / "anatomy-closeup-light.png", Inches(0.5), Inches(1.6), Inches(7.6), Inches(5.25), border=True)
    picture(s, FIG / "anatomy-exploded-light.png", Inches(8.4), Inches(1.6), Inches(4.5), Inches(2.6), border=True)
    bullets(s, Inches(8.4), Inches(4.4), Inches(4.5), Inches(2.5),
            ["Lens + IR-cut -> CMOS sensor|raw frames on one ribbon cable",
             "NPU on the main board|masks the frame in its buffer (black cells)",
             "Encoder, microSD, PoE|downstream of the mask: no raw path out"], 13, INK, 8)


def s05_data():
    s = prs.slides.add_slide(BLANK)
    header(s, 5, "Data", "Pre-trained models + openly licensed test images")
    card(s, Inches(0.6), Inches(1.75), Inches(5.9), Inches(2.4), "What the models learned from",
         ["COCO 2017|EfficientDet-Lite0 (cell phone, remote)", "Google pose / hand datasets|MediaPipe",
          "NudeNet dataset|18 body-part classes"], AMBER)
    card(s, Inches(0.6), Inches(4.35), Inches(5.9), Inches(2.4), "What the system processes",
         ["Input|640x480 colour frames", "Intermediate|33 body + 21 hand keypoints, boxes",
          "Output|masked frames + metadata only"], NAVY)
    rows = [("A", "8", "hand operating a phone / POS", TEAL), ("B", "26", "device visible, no PIN entry", AMBER),
            ("C", "21", "ordinary clothed people", ROSE)]
    text(s, Inches(6.9), Inches(1.75), Inches(5.9), Inches(0.5), "Hand-labelled evaluation set (Wikimedia Commons)",
         17, NAVY, True)
    for i, (k, n, d, c) in enumerate(rows):
        y = Inches(2.4) + i * Inches(0.95)
        rect(s, Inches(6.9), y, Inches(0.8), Inches(0.8), c)
        text(s, Inches(6.9), y, Inches(0.8), Inches(0.8), k, 24, WHITE, True, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        text(s, Inches(7.9), y + Inches(0.05), Inches(4.8), Inches(0.4), f"{n} images", 16, NAVY, True)
        text(s, Inches(7.9), y + Inches(0.42), Inches(4.8), Inches(0.4), d, 13, MUTED)
    text(s, Inches(6.9), Inches(5.35), Inches(5.9), Inches(1.4),
         "No nudity and no real PINs were collected: that would breach the privacy being protected. "
         "The webcam app records your own consented test clips with frame-level ground truth.", 13, MUTED)


def s06_techniques():
    s = prs.slides.add_slide(BLANK)
    header(s, 6, "Techniques and algorithms", "Four pre-trained networks + fusion logic built for privacy")
    models = [("Pose", "MediaPipe Pose (BlazePose)", "33 keypoints"),
              ("Body parts", "NudeNet v3 (YOLOv8n, ONNX)", "18 classes"),
              ("Devices", "EfficientDet-Lite0 int8", "COCO boxes"),
              ("Hands", "MediaPipe Hands (BlazePalm)", "21 keypoints")]
    for i, (a, b, c) in enumerate(models):
        y = Inches(1.8) + i * Inches(1.18)
        rect(s, Inches(0.6), y, Inches(5.2), Inches(1.0), PAPER)
        text(s, Inches(0.8), y + Inches(0.12), Inches(1.6), Inches(0.4), a, 15, TEAL, True)
        text(s, Inches(2.3), y + Inches(0.12), Inches(3.4), Inches(0.4), b, 14, NAVY, True)
        text(s, Inches(2.3), y + Inches(0.52), Inches(3.4), Inches(0.4), c, 12, MUTED)
    bullets(s, Inches(6.3), Inches(1.75), Inches(6.5), Inches(5.2),
            ["Pose-guided ROI crops|small people seen at full detector resolution",
             "Region-aware threshold|0.25 inside chest/pelvis region, 0.45 elsewhere",
             "Vetoes|reject parts on hands and anatomically impossible parts",
             "Pose-anchored masks|follow the body when the detector misses",
             "PIN rule|fingertip on device face (+ pointing, thumb, tapping cues)",
             "Guided zoom passes|device->hand, wrist->hand, hand->device",
             "Call filter + occlusion memory|ignore phone at ear; carry hidden phone",
             "Temporal hold + fail-closed|mask now, release after 12 empty frames",
             "Parallel branches|pose->parts, devices, hands run concurrently"], 14, INK, 6)


def s07_workflow():
    s = prs.slides.add_slide(BLANK)
    header(s, 7, "Working process", "From sensor to storage: one frame through the pipeline")
    picture(s, FIG / "pipeline_flow.png", Inches(0.5), Inches(1.7), Inches(12.3), Inches(4.7))
    text(s, Inches(0.6), Inches(6.4), Inches(12), Inches(0.5),
         "Dotted links: pose wrists and heads guide the PIN guard; tracked hands veto false body-part detections.",
         13, MUTED)


def s08_demo():
    s = prs.slides.add_slide(BLANK)
    header(s, 8, "Demo: Jupyter notebook", "Model view vs. camera output")
    picture(s, shot(11), Inches(0.5), Inches(1.55), Inches(6.3), Inches(5.35), border=True)
    text(s, Inches(7.1), Inches(1.6), Inches(5.8), Inches(0.4), "PIN guard (cell 6, left)", 13, TEAL, True)
    bullets(s, Inches(7.1), Inches(2.0), Inches(5.8), Inches(1.3),
            ["Hands + phone found|fingertip on the screen = PIN entry",
             "Output|phone and hand filled solid; rest of frame untouched"], 12, INK, 4)
    text(s, Inches(7.1), Inches(3.3), Inches(5.8), Inches(0.4), "Body privacy (pose-gated masking demo)", 13, ROSE, True)
    picture(s, shot(9, "out"), Inches(7.1), Inches(3.7), Inches(5.8), Inches(3.2), border=True)


def s09_demo2():
    s = prs.slides.add_slide(BLANK)
    header(s, 9, "Demo: robustness", "Hidden devices, permanent masks, fail-closed")
    text(s, Inches(0.5), Inches(1.55), Inches(6.2), Inches(0.4), "Device hidden by the hand: mask follows the hand",
         13, TEAL, True)
    picture(s, shot(13, "out"), Inches(0.5), Inches(1.95), Inches(6.2), Inches(2.5), border=True)
    text(s, Inches(0.5), Inches(4.6), Inches(6.2), Inches(0.4), "Blur is reversible; pixelate / solid are not",
         13, AMBER, True)
    picture(s, shot(15, "out"), Inches(0.5), Inches(5.0), Inches(6.2), Inches(1.9), border=True)
    text(s, Inches(6.95), Inches(1.55), Inches(5.9), Inches(0.4), "Simulated NPU fault: frame withheld", 13, ROSE, True)
    picture(s, shot(17), Inches(6.95), Inches(1.95), Inches(5.9), Inches(4.95), border=True)


def s10_results():
    s = prs.slides.add_slide(BLANK)
    header(s, 10, "Results", "Accurate when the detectors see the hand and device")
    vals = [(EV["Set A: sensitive regions protected (>=90% masked)"], "PIN-entry regions fully protected", TEAL),
            (EV["Set A: mean coverage of sensitive regions"], "mean coverage of sensitive regions", TEAL),
            (EV["Set B: false alarms (device, no PIN entry)"], "false alarms: idle devices", AMBER),
            (EV["Set C: false alarms (clothed people)"], "false alarms: clothed people", ROSE)]
    for i, (v, l, c) in enumerate(vals):
        stat(s, Inches(0.6) + i * Inches(3.08), Inches(1.7), Inches(2.9), v, l, c)
    picture(s, FIG / "eval_bars.png", Inches(0.5), Inches(3.25), Inches(7.6), Inches(3.6))
    bullets(s, Inches(8.4), Inches(3.4), Inches(4.4), Inches(3.5),
            ["Every miss is a detection miss|hand or phone not found (tiny, B&W, through glass)",
             "Fusion works|when both are found, the region is masked",
             "Remaining false alarm|a QR-code stand read as a phone",
             "Bank policy|masks all devices: higher recall, more masking"], 13, INK, 8)


def s11_speed():
    s = prs.slides.add_slide(BLANK)
    header(s, 11, "Speed, temporal hold, trade-offs", f"{LAT['fps']} fps on a laptop CPU; hold removes flicker leaks")
    picture(s, FIG / "latency.png", Inches(0.5), Inches(1.6), Inches(6.3), Inches(2.6))
    picture(s, FIG / "hold_curve.png", Inches(6.9), Inches(1.6), Inches(6.0), Inches(2.6))
    card(s, Inches(0.6), Inches(4.45), Inches(3.95), Inches(2.4), "Advantages",
         ["No raw footage exists", "Evidence kept", "Few false alarms", "Fail-safe, low cost"], TEAL)
    card(s, Inches(4.7), Inches(4.45), Inches(3.95), Inches(2.4), "Limitations",
         ["Small test set (8 positives)", "Tiny subjects missed", "No POS-specific class",
          f"{LAT['fps']} fps without NPU"], AMBER)
    card(s, Inches(8.8), Inches(4.45), Inches(3.95), Inches(2.4), "Ethics",
         ["DPDP Act 2023: data minimisation", "Fairness audit across skin tones", "No identification / function creep",
          "Signed firmware, no raw preview"], ROSE)


def s12_conclusion():
    s = prs.slides.add_slide(BLANK)
    header(s, 12, "Future scope and conclusion", "The camera's own AI can protect privacy instead of eroding it")
    card(s, Inches(0.6), Inches(1.75), Inches(5.6), Inches(4.9), "Future scope",
         ["Fine-tuned POS / PIN-pad detector", "Multi-person pose for crowded corridors",
          "int8 deployment on camera NPUs (RV1126, Hailo-8L, CVflow)", "Sequence model for typing (TCN / LSTM)",
          "Segmentation masks instead of boxes", "Consented, diverse video benchmark + fairness metrics"], TEAL)
    text(s, Inches(6.6), Inches(1.85), Inches(6.2), Inches(4.8),
         ["Pose estimation + body-part detection mask intimate regions; hand tracking + device detection mask "
          "PIN entry.", "",
          "Masks are permanent and applied before encoding, so unmasked footage never exists.", "",
          f"Prototype: {EV['Set A: sensitive regions protected (>=90% masked)']} PIN regions protected, "
          f"{EV['Set B: false alarms (device, no PIN entry)']} and {EV['Set C: false alarms (clothed people)']} "
          f"false alarms, {LAT['fps']} fps on CPU.", "",
          "Next: better small-object detection and an NPU for real-time speed. The architecture stays the same."],
         17, NAVY)


def main():
    for fn in (s01_title, s02_problem, s03_approach, s04_anatomy, s05_data, s06_techniques, s07_workflow,
               s08_demo, s09_demo2, s10_results, s11_speed, s12_conclusion):
        fn()
    prs.core_properties.author = AUTHOR
    prs.core_properties.last_modified_by = AUTHOR
    prs.core_properties.title = "Privacy-by-Design AI CCTV"
    out = DOCS / f"Privacy_CCTV_Slides_{AUTHOR.replace(' ', '_')}.pptx"
    prs.save(out)
    print(out)


if __name__ == "__main__":
    main()
