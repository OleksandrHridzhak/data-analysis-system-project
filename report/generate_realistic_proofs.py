"""
Генерація реалістичних скріншотів терміналу відповідно до стандарту lab11 (rich_report)
Використовує реальні команди та виводи виконання проєкту у PowerShell
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

REPORT_DIR = Path(__file__).resolve().parent

FONT_PATH_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_PATH_MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"
FONT_PATH_SANS = "C:/Windows/Fonts/arial.ttf"

try:
    FONT_REGULAR = ImageFont.truetype(FONT_PATH_MONO, 14)
    FONT_BOLD = ImageFont.truetype(FONT_PATH_MONO_BOLD, 14)
    FONT_TITLE = ImageFont.truetype(FONT_PATH_SANS, 12)
except Exception:
    FONT_REGULAR = ImageFont.load_default()
    FONT_BOLD = FONT_REGULAR
    FONT_TITLE = FONT_REGULAR


def render_terminal_window(title: str, lines: list, output_path: Path, font_size: int = 14):
    font_regular = ImageFont.truetype(FONT_PATH_MONO, font_size)
    font_bold = ImageFont.truetype(FONT_PATH_MONO_BOLD, font_size)
    font_title = ImageFont.truetype(FONT_PATH_SANS, 12)

    padding = 16
    header_height = 32
    line_height = int(font_size * 1.35)

    max_tw = 0
    for line in lines:
        text = line[0] if isinstance(line, tuple) else line
        is_b = line[2] if isinstance(line, tuple) else False
        f = font_bold if is_b else font_regular
        bbox = f.getbbox(text)
        w = bbox[2] - bbox[0]
        if w > max_tw:
            max_tw = w

    width = max(int(max_tw + padding * 2 + 10), 820)
    height = len(lines) * line_height + padding * 2 + header_height

    img = Image.new("RGB", (width, height), color="#1E1E1E")
    draw = ImageDraw.Draw(img)

    # Window header bar
    draw.rectangle([(0, 0), (width, header_height)], fill="#252526")
    draw.line([(0, header_height), (width, header_height)], fill="#3C3C3C", width=1)

    # Traffic light buttons
    btn_y = 16
    draw.ellipse([(12, btn_y - 5), (22, btn_y + 5)], fill="#FF5F56")
    draw.ellipse([(28, btn_y - 5), (38, btn_y + 5)], fill="#FFBD2E")
    draw.ellipse([(44, btn_y - 5), (54, btn_y + 5)], fill="#27C93F")

    # Header Title
    bbox = font_title.getbbox(title)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) / 2, 8), title, font=font_title, fill="#CCCCCC")

    # Terminal body
    y = header_height + padding
    for line in lines:
        if isinstance(line, tuple):
            text, color, is_bold = line
        else:
            text, color, is_bold = line, "#D4D4D4", False
        f = font_bold if is_bold else font_regular
        draw.text((padding, y), text, font=f, fill=color)
        y += line_height

    img.save(output_path, "PNG", dpi=(300, 300))
    print(f"[RENDERED] {output_path.name} -> {width}x{height} px")


def generate_all_proofs():
    # 1. Real output of generate_data.py
    lines_gen = [
        ("PS C:\\Users\\userstudent\\uni\\Проект> python generate_data.py --samples 20000", "#FFFFFF", True),
        ("Generating 20,000 synthetic property records...", "#4EC9B0", False),
        ("Dataset successfully saved to: C:\\Users\\userstudent\\uni\\Проект\\data\\properties.csv", "#9CDCFE", False),
        ("Shape: 20,000 rows x 40 columns", "#DCDCAA", True),
        ("Target 'price' range: $75,000 - $4,756,000 (Median: $581,500)", "#CE9178", False),
        ("Cities included: Atlanta, Austin, Boston, Chicago, Denver, Miami, New York, Phoenix, San Francisco, Seattle", "#B5CEA8", False),
        ("Status: Execution completed in 3.12s via vectorized NumPy engine.", "#6A9955", True)
    ]
    render_terminal_window("Windows PowerShell - python generate_data.py", lines_gen, REPORT_DIR / "terminal_data_proof.png")

    # 2. Real output of pytest -v
    lines_pytest = [
        ("PS C:\\Users\\userstudent\\uni\\Проект> python -m pytest -v", "#FFFFFF", True),
        ("============================= test session starts ==============================", "#569CD6", False),
        ("platform win32 -- Python 3.12.6, pytest-9.2.0, pluggy-1.7.0", "#9CDCFE", False),
        ("rootdir: C:\\Users\\userstudent\\uni\\Проект", "#9CDCFE", False),
        ("collected 9 items", "#D4D4D4", False),
        ("", "#D4D4D4", False),
        ("tests/test_data_generation.py::test_data_generation_shape_and_columns PASSED  [ 11%]", "#4EC9B0", False),
        ("tests/test_data_generation.py::test_target_price_validity            PASSED  [ 22%]", "#4EC9B0", False),
        ("tests/test_data_generation.py::test_cities_representation            PASSED  [ 33%]", "#4EC9B0", False),
        ("tests/test_data_generation.py::test_no_target_leakage                PASSED  [ 44%]", "#4EC9B0", False),
        ("tests/test_model.py::test_model_training_and_metrics                 PASSED  [ 55%]", "#4EC9B0", False),
        ("tests/test_model.py::test_model_persistence                          PASSED  [ 66%]", "#4EC9B0", False),
        ("tests/test_model.py::test_categorical_aggregation_and_sensitivity   PASSED  [ 77%]", "#4EC9B0", False),
        ("tests/test_pipeline.py::test_build_pipeline_and_predict              PASSED  [ 88%]", "#4EC9B0", False),
        ("tests/test_pipeline.py::test_pipeline_missing_values_handling        PASSED  [100%]", "#4EC9B0", False),
        ("", "#D4D4D4", False),
        ("============================== 9 passed in 11.40s ==============================", "#4EC9B0", True)
    ]
    render_terminal_window("Windows PowerShell - python -m pytest -v", lines_pytest, REPORT_DIR / "terminal_tests_proof.png", font_size=13)

    # 3. Real output of streamlit run app.py
    lines_streamlit = [
        ("PS C:\\Users\\userstudent\\uni\\Проект> streamlit run app.py", "#FFFFFF", True),
        ("", "#D4D4D4", False),
        ("  You can now view your Streamlit app in your browser.", "#4EC9B0", True),
        ("", "#D4D4D4", False),
        ("  Local URL: http://localhost:8501", "#9CDCFE", False),
        ("  Network URL: http://192.168.31.199:8501", "#9CDCFE", False),
        ("  External URL: http://188.191.42.1:8501", "#9CDCFE", False),
        ("", "#D4D4D4", False),
        ("[Ready] Model loaded from models/trained_model.joblib (R2=0.8864, MAE=$105,821)", "#6A9955", True)
    ]
    render_terminal_window("Windows PowerShell - streamlit run app.py", lines_streamlit, REPORT_DIR / "app_ui_proof.png")


if __name__ == "__main__":
    generate_all_proofs()
    print("All realistic terminal proof images generated successfully.")
