import json
from typing import List
import pandas as pd
from pandas.core.series import Series

README_PATH = "README.md"

PURE_CLIP_TABLE_TOKEN = "<!--- PURE_CLIP_TABLE -->"
CLIP_EXTENSIONS_TABLE_TOKEN = "<!--- CLIP_PLUS_TABLE -->"
AUDIO_TABLE_TOKEN = "<!--- AUDIO_TABLE -->"
VIDEO_TABLE_TOKEN = "<!--- VIDEO_TABLE -->"
PCD_TABLE_TOKEN = "<!--- 3D_TABLE -->"


PURE_CLIP_MODELS_FILE = "data/pure_clip_models.csv"
CLIP_EXTENSIONS_MODELS_FILE = "data/clip_extensions.csv"
AUDIO_MODELS_FILE = "data/audio_clip_models.csv"
VIDEO_MODELS_FILE = "data/video_clip_models.csv"
PCD_MODELS_FILE = "data/point_cloud_clip_models.csv"

PURE_CLIP_TABLE_HEADER = [
    "| **Model** | **Year** | **Month** | **Paper Title** | **Novel Development** | **Arxiv** | **Github** | **Open Source** | **License** | **Model Card** | **OpenCLIP Integration** |",
    "|:---------:|:---------:|:--------:|:----------------:|:----------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|",
]

CLIP_EXTENSIONS_TABLE_HEADER = [
    "| **Model** | **Year** | **Month** | **Paper Title** | **Pretraining Techniques** | **Arxiv** | **Github** | **Open Source** | **License** |",
    "|:---------:|:---------:|:--------:|:----------------:|:----------------:|:--------:|:--------:|:--------:|:--------:|",
]


AUDIO_TABLE_HEADER = [
    "| **Model** | **Year** | **Month** | **Paper Title** | **Modalities** | **Arxiv** | **Github** | **Open Source** | **License** |",
    "|:---------:|:---------:|:--------:|:----------------:|:----------------:|:--------:|:--------:|:--------:|:--------:|",
]

VIDEO_TABLE_HEADER = [
    "| **Model** | **Year** | **Month** | **Paper Title** | **Arxiv** | **Github** | **Open Source** | **License** |",
    "|:---------:|:---------:|:--------:|:----------------:|:--------:|:--------:|:--------:|:--------:|",
]

PCD_TABLE_HEADER = [
    "| **Model** | **Year** | **Month** | **Paper Title** | **Modalities** | **Arxiv** | **Github** | **Open Source** | **License** |",
    "|:---------:|:---------:|:--------:|:----------------:|:--------:|:--------:|:--------:|:--------:|:--------:|",
]



GITHUB_PREFIX = "https://github.com/"
GITHUB_BADGE_PATTERN = (
    "[![GitHub](https://img.shields.io/github/stars/{}?style=social)]({})"
)

ARXIV_BADGE_PATTERN = "[![arXiv](https://img.shields.io/badge/arXiv-{}-b31b1b.svg)](https://arxiv.org/abs/{})"
LICENSE_PATTERN = "[License]({})"
MODEL_CARD_PATTERN = "[Model Card]({})"

def _get_license(entry):
    license = entry["license"] if type(entry["license"]) == str else ""
    license_entry = LICENSE_PATTERN.format(license) if license else ""
    return license_entry

def _get_model_card(entry):
    model_card = entry["model_card"] if type(entry["model_card"]) == str else ""
    model_card_entry = MODEL_CARD_PATTERN.format(model_card) if model_card else ""
    return model_card_entry

def _get_openclip_integration(entry):
    openclip_integration = entry["open_clip_integration"]
    open_clip_entry = "✔️" if openclip_integration else "❌"
    return open_clip_entry

def _get_open_source(entry):
    open_source = (entry["open_source"] == "Y")
    open_source_entry = "✔️" if open_source else "❌"
    return open_source_entry

def _get_github_badge(entry):
    github = entry["github"]
    github_badge = (
        GITHUB_BADGE_PATTERN.format("/".join(github.split("/")[0:2]), GITHUB_PREFIX + github)
        if type(github) == str
        else ""
    )
    return github_badge

def _get_arxiv_badge(entry):
    arxiv = entry["arxiv"]
    arxiv_badge = ARXIV_BADGE_PATTERN.format(arxiv, arxiv)
    return arxiv_badge


def _get_basics(entry):
    model_name = entry["model_name"]
    year = entry["year"]
    month = entry["month"]
    paper_title = entry["title"]
    return model_name, year, month, paper_title


def _get_universals(entry):
    github_badge = _get_github_badge(entry)
    arxiv_badge = _get_arxiv_badge(entry)
    open_source_entry = _get_open_source(entry)
    license_entry = _get_license(entry)
    return github_badge, arxiv_badge, open_source_entry, license_entry


def _get_pure_clip_row(entry):
    model_name, year, month, paper_title = _get_basics(entry)
    github_badge, arxiv_badge, open_source_entry, license_entry = _get_universals(entry)
    openclip_integration = _get_openclip_integration(entry)
    development = entry["development"]
    model_card_entry = _get_model_card(entry)
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        model_name,
        year,
        month,
        paper_title,
        development,
        arxiv_badge,
        github_badge,
        open_source_entry,
        license_entry,
        model_card_entry,
        openclip_integration,
    )

def _get_clip_plus_row(entry):
    model_name, year, month, paper_title = _get_basics(entry)
    github_badge, arxiv_badge, open_source_entry, license_entry = _get_universals(entry)
    pretraining = entry["pretraining_techniques"]
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        model_name,
        year,
        month,
        paper_title,
        pretraining,
        arxiv_badge,
        github_badge,
        open_source_entry,
        license_entry,
    )

def _get_audio_row(entry):
    model_name, year, month, paper_title = _get_basics(entry)
    github_badge, arxiv_badge, open_source_entry, license_entry = _get_universals(entry)
    modalities = entry["modalities"]
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        model_name,
        year,
        month,
        paper_title,
        modalities,
        arxiv_badge,
        github_badge,
        open_source_entry,
        license_entry,
    )

def _get_video_row(entry):
    model_name, year, month, paper_title = _get_basics(entry)
    github_badge, arxiv_badge, open_source_entry, license_entry = _get_universals(entry)
    return "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
        model_name,
        year,
        month,
        paper_title,
        arxiv_badge,
        github_badge,
        open_source_entry,
        license_entry,
    )

def _get_pcd_row(entry):
    model_name, year, month, paper_title = _get_basics(entry)
    github_badge, arxiv_badge, open_source_entry, license_entry = _get_universals(entry)
    modalities = entry["modalities"]
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        model_name,
        year,
        month,
        paper_title,
        modalities,
        arxiv_badge,
        github_badge,
        open_source_entry,
        license_entry,
    )

def _generate_pure_clip_table() -> List[str]:
    """Generates markdown table from data file."""

    df = pd.read_csv(PURE_CLIP_MODELS_FILE)
    table = PURE_CLIP_TABLE_HEADER.copy()
    for _, entry in df.iterrows():
        table.append(_get_pure_clip_row(entry))
    return table

def _generate_clip_plus_table() -> List[str]:
    """Generates markdown table from data file."""

    df = pd.read_csv(CLIP_EXTENSIONS_MODELS_FILE)
    table = CLIP_EXTENSIONS_TABLE_HEADER.copy()
    for _, entry in df.iterrows():
        table.append(_get_clip_plus_row(entry))
    return table

def _generate_audio_table() -> List[str]:
    """Generates markdown table from data file."""

    df = pd.read_csv(AUDIO_MODELS_FILE)
    table = AUDIO_TABLE_HEADER.copy()
    for _, entry in df.iterrows():
        table.append(_get_audio_row(entry))
    return table

def _generate_video_table() -> List[str]:
    """Generates markdown table from data file."""

    df = pd.read_csv(VIDEO_MODELS_FILE)
    table = VIDEO_TABLE_HEADER.copy()
    for _, entry in df.iterrows():
        table.append(_get_video_row(entry))
    return table

def _generate_pcd_table() -> List[str]:
    """Generates markdown table from data file."""

    df = pd.read_csv(PCD_MODELS_FILE)
    table = PCD_TABLE_HEADER.copy()
    for _, entry in df.iterrows():
        table.append(_get_pcd_row(entry))
    return table


def read_lines_from_file(path: str) -> List[str]:
    ''' Reads lines from file and strips trailing whitespaces. '''
    with open(path) as file:
        return [line.rstrip() for line in file]

def save_lines_to_file(path: str, lines: List[str]) -> None:
    ''' Saves lines to file. '''
    with open(path, "w") as f:
        for line in lines:
            f.write("%s\n" % line)


def search_lines_with_token(lines: List[str], token: str) -> List[int]:
    ''' Searches for lines with token. '''
    result = []
    for line_index, line in enumerate(lines):
        if token in line:
            result.append(line_index)
    return result


def _inject_table_into_readme(readme_lines: List[str], table_lines: List[str], token: str) -> List[str]:
    ''' Injects table into readme. '''
    lines_with_token_indexes = search_lines_with_token(lines=readme_lines, token=token)
    if len(lines_with_token_indexes) != 2:
        raise Exception(f"Please inject one {token} token to signal start of autogenerated table.")

    [table_start_line_index, table_end_line_index] = lines_with_token_indexes
    return readme_lines[:table_start_line_index + 1] + table_lines + readme_lines[table_end_line_index:]


def inject_markdown_tables_into_readme(readme_lines: List[str]) -> List[str]:
    ''' Injects markdown tables into readme. '''

    # Inject pure clip table
    pure_clip_table_lines = _generate_pure_clip_table()
    readme_lines = _inject_table_into_readme(
        readme_lines=readme_lines,
        table_lines=pure_clip_table_lines,
        token=PURE_CLIP_TABLE_TOKEN
    )

    # Inject clip plus table
    clip_plus_table_lines = _generate_clip_plus_table()
    readme_lines = _inject_table_into_readme(
        readme_lines=readme_lines,
        table_lines=clip_plus_table_lines,
        token=CLIP_EXTENSIONS_TABLE_TOKEN
    )

    # Inject audio table
    audio_table_lines = _generate_audio_table()
    readme_lines = _inject_table_into_readme(
        readme_lines=readme_lines,
        table_lines=audio_table_lines,
        token=AUDIO_TABLE_TOKEN
    )

    # Inject video table
    video_table_lines = _generate_video_table()
    readme_lines = _inject_table_into_readme(
        readme_lines=readme_lines,
        table_lines=video_table_lines,
        token=VIDEO_TABLE_TOKEN
    )

    # Inject pcd table
    pcd_table_lines = _generate_pcd_table()
    readme_lines = _inject_table_into_readme(
        readme_lines=readme_lines,
        table_lines=pcd_table_lines,
        token=PCD_TABLE_TOKEN
    )

    return readme_lines
    

def generate_readme():
    readme_path = README_PATH
    readme_lines = read_lines_from_file(readme_path)
    readme_lines = inject_markdown_tables_into_readme(readme_lines=readme_lines)
    save_lines_to_file(path=readme_path, lines=readme_lines)


def main() -> None:
    generate_readme()


if __name__ == "__main__":
    main()