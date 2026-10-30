"""Export the executed notebook to HTML and screenshot each cell (input + output) with Edge."""
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "privacy_cctv_demo.ipynb"
HTML = ROOT / "notebooks" / "privacy_cctv_demo.html"
OUT = ROOT / "docs" / "notebook_screens"


def export_html() -> None:
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--template", "lab",
                    str(NB), "--output", HTML.stem], check=True)


def capture() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    shots = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1.5)
        page.goto(HTML.as_uri())
        page.wait_for_load_state("networkidle")
        cells = page.query_selector_all(".jp-Cell")
        for i, cell in enumerate(cells):
            kind = "code" if "jp-CodeCell" in (cell.get_attribute("class") or "") else "md"
            path = OUT / f"cell_{i:02d}_{kind}.png"
            cell.scroll_into_view_if_needed()
            cell.screenshot(path=str(path))
            shots.append(path)
            output = cell.query_selector(".jp-Cell-outputWrapper")
            if kind == "code" and output and output.bounding_box() and output.bounding_box()["height"] > 20:
                output.screenshot(path=str(OUT / f"cell_{i:02d}_out.png"))
        browser.close()
    return shots


if __name__ == "__main__":
    export_html()
    print("\n".join(p.name for p in capture()))
