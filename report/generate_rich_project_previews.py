import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

REPORT_DIR = Path(__file__).resolve().parent
BASE_DIR = REPORT_DIR.parent

# Colors for VS Code style editor preview
BG_COLOR = (30, 30, 30)
BAR_COLOR = (45, 45, 48)
TAB_ACTIVE_COLOR = (30, 30, 30)
BORDER_COLOR = (60, 60, 60)
LINE_NUM_COLOR = (100, 100, 100)

TEXT_WHITE = (225, 225, 225)
TEXT_GRAY = (150, 150, 150)
TEXT_GREEN = (90, 210, 150)
TEXT_BLUE = (95, 170, 235)
TEXT_YELLOW = (235, 215, 140)
TEXT_ORANGE = (220, 150, 110)
TEXT_RED = (245, 80, 80)
TEXT_CYAN = (130, 215, 245)
TEXT_MAGENTA = (205, 145, 205)

FONT_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"
FONT_SANS = "C:/Windows/Fonts/segoeui.ttf"

try:
    FONT = ImageFont.truetype(FONT_MONO, 18)
    FONT_BOLD = ImageFont.truetype(FONT_MONO_BOLD, 18)
    FONT_HEADER = ImageFont.truetype(FONT_SANS, 15)
except Exception:
    FONT = ImageFont.load_default()
    FONT_BOLD = FONT
    FONT_HEADER = FONT


def render_editor(file_name: str, lines_tokens: list, output_path: Path, width=1350):
    line_height = 28
    pad_top = 44
    pad_bottom = 20
    pad_left = 68

    height = pad_top + len(lines_tokens) * line_height + pad_bottom
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Top tab bar
    draw.rectangle([(0, 0), (width, 36)], fill=BAR_COLOR)
    draw.line([(0, 36), (width, 36)], fill=BORDER_COLOR, width=1)

    tab_width = 380
    draw.rectangle([(16, 4), (16 + tab_width, 36)], fill=TAB_ACTIVE_COLOR)
    draw.line([(16, 4), (16 + tab_width, 4)], fill=TEXT_BLUE, width=2)
    draw.text((36, 8), file_name, fill=TEXT_WHITE, font=FONT_HEADER)

    draw.line([(56, pad_top), (56, height - pad_bottom)], fill=BORDER_COLOR, width=1)

    y = pad_top
    for i, line in enumerate(lines_tokens):
        line_num = str(i + 1)
        draw.text((46 - len(line_num) * 11, y), line_num, fill=LINE_NUM_COLOR, font=FONT)

        x = pad_left
        for item in line:
            if len(item) == 2:
                text, color = item
                is_bold = False
            else:
                text, color, is_bold = item
            f = FONT_BOLD if is_bold else FONT
            draw.text((x, y), text, fill=color, font=f)
            bbox = f.getbbox(text)
            w = bbox[2] - bbox[0]
            x += w
        y += line_height

    img.save(output_path, "PNG", dpi=(300, 300))
    print(f"[EDITOR] Saved: {output_path.name}")


# 1. Preview of data/properties.csv
def make_properties_csv_preview():
    lines = [
        [("city,state,property_type,living_area_sqft,lot_size,beds,baths,year_built,condition,price", TEXT_GRAY, True)],
        [("San Francisco,CA,Single Family,", TEXT_WHITE), ("2415", TEXT_MAGENTA), (",8500,4,3.0,1984,", TEXT_WHITE), ("Good,", TEXT_GREEN), ("1510000.0", TEXT_YELLOW, True)],
        [("Miami,FL,Single Family,", TEXT_WHITE), ("3129", TEXT_MAGENTA), (",9200,5,3.5,2002,", TEXT_WHITE), ("Good,", TEXT_GREEN), ("893000.0", TEXT_YELLOW, True)],
        [("Atlanta,GA,Single Family,", TEXT_WHITE), ("1397", TEXT_MAGENTA), (",6800,2,1.5,1976,", TEXT_WHITE), ("Fair,", TEXT_ORANGE), ("301500.0", TEXT_YELLOW, True)],
        [("Denver,CO,Townhouse,", TEXT_WHITE), ("2169", TEXT_MAGENTA), (",2400,3,3.0,2012,", TEXT_WHITE), ("Excellent,", TEXT_BLUE), ("529000.0", TEXT_YELLOW, True)],
        [("San Francisco,CA,Condo,", TEXT_WHITE), ("852", TEXT_MAGENTA), (",0,1,1.0,2008,", TEXT_WHITE), ("Good,", TEXT_GREEN), ("599500.0", TEXT_YELLOW, True)],
        [("Phoenix,AZ,Single Family,", TEXT_WHITE), ("2828", TEXT_MAGENTA), (",11500,4,3.0,1995,", TEXT_WHITE), ("Excellent,", TEXT_BLUE), ("888000.0", TEXT_YELLOW, True)],
        [("Seattle,WA,Single Family,", TEXT_WHITE), ("2640", TEXT_MAGENTA), (",7800,4,2.5,2015,", TEXT_WHITE), ("Luxury,", TEXT_MAGENTA), ("1245000.0", TEXT_YELLOW, True)],
        [("Chicago,IL,Single Family,", TEXT_WHITE), ("1920", TEXT_MAGENTA), (",6200,3,2.0,1968,", TEXT_WHITE), ("Good,", TEXT_GREEN), ("475000.0", TEXT_YELLOW, True)],
        [("New York,NY,Condo,", TEXT_WHITE), ("1210", TEXT_MAGENTA), (",0,2,2.0,2018,", TEXT_WHITE), ("Luxury,", TEXT_MAGENTA), ("1380000.0", TEXT_YELLOW, True)]
    ]
    render_editor("data/properties.csv (Top 10 of 20,000 records)", lines, REPORT_DIR / "data_preview_properties_csv.png")


# 2. Preview of src/pipeline.py
def make_pipeline_code_preview():
    lines = [
        [("def ", TEXT_BLUE, True), ("build_pipeline", TEXT_YELLOW, True), ("(numerical_features, categorical_features, ...):", TEXT_WHITE)],
        [("    num_transformer = Pipeline(steps=[", TEXT_WHITE)],
        [("        (", TEXT_WHITE), ("'imputer'", TEXT_ORANGE), (", SimpleImputer(strategy=", TEXT_WHITE), ("'median'", TEXT_GREEN), (")),", TEXT_WHITE)],
        [("        (", TEXT_WHITE), ("'scaler'", TEXT_ORANGE), (", StandardScaler())", TEXT_WHITE)],
        [("    ])", TEXT_WHITE)],
        [("    cat_transformer = Pipeline(steps=[", TEXT_WHITE)],
        [("        (", TEXT_WHITE), ("'imputer'", TEXT_ORANGE), (", SimpleImputer(strategy=", TEXT_WHITE), ("'constant'", TEXT_GREEN), (", fill_value=", TEXT_WHITE), ("'missing'", TEXT_GREEN), (")),", TEXT_WHITE)],
        [("        (", TEXT_WHITE), ("'onehot'", TEXT_ORANGE), (", OneHotEncoder(handle_unknown=", TEXT_WHITE), ("'ignore'", TEXT_GREEN), (", sparse_output=False))", TEXT_WHITE)],
        [("    ])", TEXT_WHITE)],
        [("    preprocessor = ColumnTransformer(transformers=[", TEXT_WHITE)],
        [("        (", TEXT_WHITE), ("'num'", TEXT_ORANGE), (", num_transformer, numerical_features),", TEXT_WHITE)],
        [("        (", TEXT_WHITE), ("'cat'", TEXT_ORANGE), (", cat_transformer, categorical_features)", TEXT_WHITE)],
        [("    ])", TEXT_WHITE)],
        [("    regressor = RandomForestRegressor(n_estimators=80, max_depth=14, n_jobs=-1)", TEXT_CYAN, True)],
        [("    return ", TEXT_BLUE, True), ("Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])", TEXT_WHITE)]
    ]
    render_editor("src/pipeline.py (ColumnTransformer & RandomForestRegressor)", lines, REPORT_DIR / "data_preview_pipeline_py.png")


# 3. Preview of exported predictions batch CSV
def make_predictions_csv_preview():
    lines = [
        [("city,property_type,living_sqft,Actual_Price,Predicted_Price,Absolute_Error,Percentage_Error", TEXT_GRAY, True)],
        [("Austin,Single Family,2350,", TEXT_WHITE), ("540000.0,", TEXT_YELLOW), ("538200.0,", TEXT_CYAN), ("1800.0,", TEXT_GREEN), ("0.33%", TEXT_GREEN, True)],
        [("San Francisco,Single Family,2800,", TEXT_WHITE), ("1580000.0,", TEXT_YELLOW), ("1542100.0,", TEXT_CYAN), ("37900.0,", TEXT_GREEN), ("2.40%", TEXT_GREEN, True)],
        [("Seattle,Single Family,2100,", TEXT_WHITE), ("820000.0,", TEXT_YELLOW), ("808500.0,", TEXT_CYAN), ("11500.0,", TEXT_GREEN), ("1.40%", TEXT_GREEN, True)],
        [("Miami,Condo,950,", TEXT_WHITE), ("450000.0,", TEXT_YELLOW), ("462800.0,", TEXT_CYAN), ("12800.0,", TEXT_GREEN), ("2.84%", TEXT_GREEN, True)],
        [("Chicago,Single Family,1850,", TEXT_WHITE), ("460000.0,", TEXT_YELLOW), ("471200.0,", TEXT_CYAN), ("11200.0,", TEXT_GREEN), ("2.43%", TEXT_GREEN, True)],
        [("Denver,Townhouse,1750,", TEXT_WHITE), ("510000.0,", TEXT_YELLOW), ("498900.0,", TEXT_CYAN), ("11100.0,", TEXT_GREEN), ("2.18%", TEXT_GREEN, True)],
        [("New York,Condo,1100,", TEXT_WHITE), ("1150000.0,", TEXT_YELLOW), ("1178400.0,", TEXT_CYAN), ("28400.0,", TEXT_GREEN), ("2.47%", TEXT_GREEN, True)],
        [("Phoenix,Single Family,2200,", TEXT_WHITE), ("420000.0,", TEXT_YELLOW), ("414200.0,", TEXT_CYAN), ("5800.0,", TEXT_GREEN), ("1.38%", TEXT_GREEN, True)]
    ]
    render_editor("real_estate_predictions.csv (Exported Batch Inference Batch)", lines, REPORT_DIR / "data_preview_predictions_csv.png")


# 4. Preview of Project Directory Tree
def make_tree_preview():
    lines = [
        [("c:\\Users\\userstudent\\uni\\Проект\\", TEXT_CYAN, True)],
        [("├── data/properties.csv               ", TEXT_GREEN), ("# 20,000 синтетичних записів (40 колонок)", TEXT_GRAY)],
        [("├── models/trained_model.joblib      ", TEXT_GREEN), ("# Серіалізований пайплайн RandomForest (R2=0.8864)", TEXT_GRAY)],
        [("├── src/                             ", TEXT_YELLOW, True), ("# Модульна архітектура бекенду та UI", TEXT_GRAY)],
        [("│   ├── config.py                    ", TEXT_WHITE), ("# Константи, профілі 10 міст США, списки фіч", TEXT_GRAY)],
        [("│   ├── data_generator.py            ", TEXT_WHITE), ("# Векторизована генерація масиву без leakage", TEXT_GRAY)],
        [("│   ├── pipeline.py                  ", TEXT_WHITE), ("# scikit-learn ColumnTransformer + RandomForest", TEXT_GRAY)],
        [("│   ├── model_manager.py             ", TEXT_WHITE), ("# Навчання, метрики (MAE, RMSE, R2), joblib IO", TEXT_GRAY)],
        [("│   ├── analytics.py                 ", TEXT_WHITE), ("# Агрегації за категоріями та аналіз чутливості", TEXT_GRAY)],
        [("│   └── ui/                          ", TEXT_YELLOW), ("# Компоненти користувацького інтерфейсу", TEXT_GRAY)],
        [("│       ├── components.py            ", TEXT_WHITE), ("# Інтерактивні Plotly графіки та картки", TEXT_GRAY)],
        [("│       ├── training_view.py         ", TEXT_WHITE), ("# Вкладка тренування моделі та гіперпараметрів", TEXT_GRAY)],
        [("│       └── prediction_view.py       ", TEXT_WHITE), ("# Вкладка одиничного/пакетного прогнозування", TEXT_GRAY)],
        [("├── tests/                           ", TEXT_YELLOW, True), ("# Набір автоматизованих тестів pytest", TEXT_GRAY)],
        [("│   ├── test_data_generation.py      ", TEXT_WHITE), ("# Перевірка форми даних, валідності цін та витоку", TEXT_GRAY)],
        [("│   ├── test_pipeline.py             ", TEXT_WHITE), ("# Перевірка SimpleImputer та OneHotEncoder", TEXT_GRAY)],
        [("│   └── test_model.py                ", TEXT_WHITE), ("# Перевірка навчання, персистентності та метрик", TEXT_GRAY)],
        [("├── generate_data.py                 ", TEXT_CYAN), ("# Автономний консольний скрипт генерації", TEXT_GRAY)],
        [("├── app.py                           ", TEXT_CYAN), ("# Точка входу вебдодатку Streamlit", TEXT_GRAY)],
        [("├── requirements.txt                 ", TEXT_WHITE), ("# Залежності середовища (Python 3.12+)", TEXT_GRAY)],
        [("└── README.md                        ", TEXT_WHITE), ("# Повна інструкція та опис архітектури", TEXT_GRAY)]
    ]
    render_editor("Project Architecture Tree (c:\\Users\\userstudent\\uni\\Проект)", lines, REPORT_DIR / "data_preview_structure_tree.png")


# 5. Generate high-quality scientific plots
def make_scientific_plots():
    df = pd.read_csv(BASE_DIR / "data" / "properties.csv")
    bundle = joblib.load(BASE_DIR / "models" / "trained_model.joblib")
    pipeline = bundle["pipeline"]
    
    # 5.1 Actual vs Predicted Plot
    eval_sample = bundle["eval_sample"]
    sample_plt = eval_sample.sample(min(2500, len(eval_sample)), random_state=42)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=220)
    sc = ax.scatter(
        sample_plt["Actual_Price"] / 1000.0,
        sample_plt["Predicted_Price"] / 1000.0,
        c=sample_plt["Pct_Error"],
        cmap="viridis",
        alpha=0.65,
        s=16,
        edgecolors="none"
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Відносна похибка прогнозу (%)", fontsize=10)
    
    min_v = min(sample_plt["Actual_Price"].min(), sample_plt["Predicted_Price"].min()) / 1000.0
    max_v = max(sample_plt["Actual_Price"].max(), sample_plt["Predicted_Price"].max()) / 1000.0
    ax.plot([min_v, max_v], [min_v, max_v], color="#e53e3e", linestyle="--", linewidth=1.8, label="Ідеальний прогноз (y = x)")
    
    ax.set_title("Actual vs. Predicted Real Estate Prices (Тестова вибірка 2500 об'єктів)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Фактична ціна житла ($ тис.)", fontsize=10)
    ax.set_ylabel("Змодельована ціна житла ($ тис.)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "fig_actual_vs_predicted.png", dpi=250)
    plt.close(fig)
    print("[PLOT] Saved: fig_actual_vs_predicted.png")

    # 5.2 Feature Importance Plot
    df_imp = bundle["df_importance"].head(14).sort_values(by="Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=220)
    bars = ax.barh(df_imp["Feature"], df_imp["Importance"], color="#3182ce", edgecolor="black", linewidth=0.6)
    ax.set_title("Важливість ознак моделі RandomForestRegressor (MDI Score)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Оцінка відносної важливості (Gini Importance)", fontsize=10)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, max(df_imp["Importance"]) * 1.15)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "fig_feature_importance.png", dpi=250)
    plt.close(fig)
    print("[PLOT] Saved: fig_feature_importance.png")

    # 5.3 Sensitivity Analysis Plot
    base_prop = eval_sample.iloc[0].to_dict()
    living_grid = np.linspace(800, 6000, 40)
    test_batch = []
    for val in living_grid:
        p = dict(base_prop)
        p["living_area_sqft"] = val
        test_batch.append(p)
    preds_sens = pipeline.predict(pd.DataFrame(test_batch))
    
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=220)
    ax.plot(living_grid, preds_sens / 1000.0, color="#2b6cb0", linewidth=2.5, marker="o", markersize=4, label="Крива чутливості моделі")
    ax.scatter([base_prop["living_area_sqft"]], [pipeline.predict(pd.DataFrame([base_prop]))[0] / 1000.0], color="#e53e3e", s=90, zorder=5, label="Базовий об'єкт")
    ax.set_title("Аналіз чутливості ціни до зміни житлової площі (What-If)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Житлова площа (living_area_sqft, кв. фути)", fontsize=10)
    ax.set_ylabel("Прогнозована вартість ($ тис.)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "fig_sensitivity.png", dpi=250)
    plt.close(fig)
    print("[PLOT] Saved: fig_sensitivity.png")


if __name__ == "__main__":
    make_properties_csv_preview()
    make_pipeline_code_preview()
    make_predictions_csv_preview()
    make_tree_preview()
    make_scientific_plots()
    print("All rich previews generated.")

