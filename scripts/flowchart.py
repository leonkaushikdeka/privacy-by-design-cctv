"""Draws the per-frame workflow diagram used in the report and slides."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "pipeline_flow.png"

INK, MUTED = "#1d2433", "#5b6475"
FILL = {"io": "#e8edf5", "body": "#fde3ec", "pin": "#dff3ee", "shared": "#fff1d6", "fault": "#f6d6d6"}
EDGE = {"io": "#8a97ad", "body": "#d35d8a", "pin": "#2a9d8f", "shared": "#d99a1f", "fault": "#b54141"}


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=10, color=INK, weight="bold")
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.8, color=MUTED, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, a, b, color=INK, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, lw=1.3, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def main():
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 0.62)
    ax.axis("off")
    w, h = 0.19, 0.11

    sensor = box(ax, 0.008, 0.255, 0.145, h, "Sensor frame", "raw, inside SoC only", "io")

    pose = box(ax, 0.2, 0.44, w, h, "Pose estimation", "MediaPipe Pose\n33 keypoints / person", "body")
    parts = box(ax, 0.43, 0.44, w, h, "Body-part detector", "NudeNet YOLOv8n (ONNX)\nfull frame + pose ROI crops", "body")
    gate = box(ax, 0.66, 0.44, w, h, "Pose gating + vetoes", "region-aware threshold\nhand veto, anatomy check", "body")

    dev = box(ax, 0.2, 0.07, w, h, "Device detector", "EfficientDet-Lite0\nphone / POS (COCO)", "pin")
    hands = box(ax, 0.43, 0.07, w, h, "Hand tracking", "MediaPipe Hands, 21 pts\n+ 3 guided zoom passes", "pin")
    rule = box(ax, 0.66, 0.07, w, h, "PIN-entry rule", "fingertip on device face\ncall filter, occlusion memory", "pin")

    hold = box(ax, 0.92, 0.255, 0.19, h, "Temporal hold", "mask at once, release\nafter 12 empty frames", "shared")
    mask = box(ax, 1.15, 0.255, 0.19, h, "Irreversible mask", "in-place: solid (PIN)\npixelate (body)", "shared")
    enc = box(ax, 1.39, 0.255, 0.21, h, "Encoder -> SD / LAN", "only masked frames\never reach this stage", "io")
    fault = box(ax, 1.15, 0.46, 0.19, 0.1, "Fail closed", "model error ->\nblank frame", "fault")

    arrow(ax, right(sensor), left(pose), rad=-0.15)
    arrow(ax, right(sensor), left(dev), rad=0.15)
    arrow(ax, right(pose), left(parts))
    arrow(ax, right(parts), left(gate))
    arrow(ax, right(dev), left(hands))
    arrow(ax, right(hands), left(rule))
    arrow(ax, right(gate), left(hold), rad=-0.12)
    arrow(ax, right(rule), left(hold), rad=0.12)
    arrow(ax, right(hold), left(mask))
    arrow(ax, right(mask), left(enc))
    arrow(ax, (fault[0] + fault[2] / 2, fault[1]), (mask[0] + mask[2] / 2, mask[1] + mask[3]), color=EDGE["fault"],
          ls="--")
    # cross-pipeline links: pose wrists/heads -> PIN guard, hands -> body veto
    arrow(ax, (pose[0] + w * 0.5, pose[1]), (hands[0] + w * 0.2, hands[1] + h), color=MUTED, ls=":", rad=0.0)
    arrow(ax, (hands[0] + w * 0.8, hands[1] + h), (gate[0] + w * 0.3, gate[1]), color=MUTED, ls=":", rad=0.0)
    ax.text(0.33, 0.30, "wrists,\nheads", fontsize=7.5, color=MUTED, ha="right")
    ax.text(0.69, 0.30, "hands\n(veto)", fontsize=7.5, color=MUTED)

    ax.text(0.20, 0.585, "BODY PRIVACY", fontsize=9, color=EDGE["body"], weight="bold")
    ax.text(0.20, 0.035, "PIN GUARD", fontsize=9, color=EDGE["pin"], weight="bold")
    ax.text(0.92, 0.585, "ON-CAMERA NPU BOUNDARY", fontsize=9, color=MUTED, weight="bold")
    ax.plot([0.175, 1.36, 1.36, 0.175, 0.175], [0.02, 0.02, 0.605, 0.605, 0.02], color=MUTED, lw=1, ls="--")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
