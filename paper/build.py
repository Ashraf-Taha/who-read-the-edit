"""Assemble paper/draft.md from paper/src/*.md and the generated numbers.

    python paper/make_numbers.py && python paper/build.py

{{name}} is replaced by values.json[name] and {{table:name}} by generated/name.md.
Tables and figures are numbered in order of appearance, each kind on its own: a caption
"**Table {{#key}}.**" or "**Figure {{#key}}.**" defines the
label `key` (keys starting with a capital letter and a colon, such as "A:", are numbered A1, A2, ...
per appendix), and
{{#key}} anywhere else is replaced by that number.
{{range:a.*.b}} gives the range of a.<model>.b over the three small models ("75% to 79%", or one value
if they are equal); {{range:a.(x|y).b}} does the same over x and y.
Numbers in scientific notation (3.4e-11) are set as $3.4 \times 10^{-11}$ in the text; tables keep
the short form.
Anything missing is marked in the draft as [MISSING ...] and listed on stdout, as are
citation keys [@key] that have no entry in paper/refs.bib. Exit status 1 if anything
is missing, so a draft with gaps is never mistaken for a finished one.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\{\{([^}]+)\}\}")
TABLE = re.compile(r"\{\{table:([^}]+)\}\}")
CAPTION = re.compile(r"\*\*(Table|Figure) \{\{#([^}]+)\}\}\.\*\*")
CITATION = re.compile(r"@([A-Za-z0-9_:-]+)")
SMALL_MODELS = ("qwen0.5b", "qwen1.5b", "smol1.7b")
SCIENTIFIC = re.compile(r"(?<![\w.$])(-?\d+(?:\.\d+)?)e([+-]?)(\d+)\b")
NUMBER = re.compile(r"^[-−+]?\d+(?:\.\d+)?(?:e[+-]?\d+)?%?$")


def scientific(text: str) -> str:
    text = SCIENTIFIC.sub(lambda m: f"${m[1]} \\times 10^{{{'-' if m[2] == '-' else ''}{int(m[3])}}}$", text)
    return re.sub(r"(?<![\w${-])-(?=\d)", "\u2212", text)


def value_range(name: str, values: dict) -> str | None:
    pattern = name[len("range:"):]
    choice = re.search(r"\(([^)]*)\)", pattern)
    if choice:
        keys = [pattern[:choice.start()] + option + pattern[choice.end():] for option in choice.group(1).split("|")]
    elif "*" in pattern:
        keys = [pattern.replace("*", model) for model in SMALL_MODELS]
    else:
        return None
    found = [values.get(key) for key in keys]
    if None not in found and len(set(found)) == 1:
        return found[0]
    counts = [re.fullmatch(r"(\d+) of (\d+)", v or "") for v in found]
    if all(counts) and len({c[2] for c in counts}) == 1:
        low, high = min(int(c[1]) for c in counts), max(int(c[1]) for c in counts)
        return f"{low} to {high} of {counts[0][2]}"
    if any(v is None or not NUMBER.match(v) for v in found):
        return None
    number = lambda v: float(v.replace("−", "-").rstrip("%"))
    low, high = min(found, key=number), max(found, key=number)
    return low if number(low) == number(high) else f"{low} to {high}"


def main() -> int:
    values = json.loads((HERE / "generated" / "values.json").read_text())
    text = "\n\n".join(p.read_text().strip() for p in sorted((HERE / "src").glob("*.md")))
    missing: list[str] = []

    def include(match: re.Match) -> str:
        path = HERE / "generated" / f"{match.group(1).strip()}.md"
        if path.exists():
            return path.read_text().strip()
        missing.append(f"table:{match.group(1).strip()}")
        return f"**[MISSING table:{match.group(1).strip()}]**"

    text = TABLE.sub(include, text)
    numbers: dict[str, str] = {}
    kinds: dict[str, str] = {}
    letter = lambda key: key[0] if len(key) > 2 and key[1] == ":" and key[0].isupper() else ""
    for kind, key in CAPTION.findall(text):
        if key in numbers:
            missing.append(f"#{key} (two captions)")
            continue
        numbers[key] = letter(key) + str(1 + sum(letter(k) == letter(key) and kinds[k] == kind for k in numbers))
        kinds[key] = kind

    def fill(match: re.Match) -> str:
        name = match.group(1).strip()
        if name.startswith("#") and name[1:] in numbers:
            return numbers[name[1:]]
        if name.startswith("range:"):
            spread = value_range(name, values)
            if spread is not None:
                return scientific(spread)
        elif name in values:
            return scientific(values[name])
        missing.append(name)
        return f"**[MISSING {name}]**"

    text = PLACEHOLDER.sub(fill, text)
    text = PLACEHOLDER.sub(fill, text)  # generated values may themselves hold cross-references
    bib = (HERE / "refs.bib").read_text() if (HERE / "refs.bib").exists() else ""
    known = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = sorted({key for block in re.findall(r"\[(@[^\]]+)\]", text) for key in CITATION.findall(block)})
    uncited = sorted(set(cited) - known)
    notes = re.findall(r"<!--(.*?)-->", text, flags=re.S)

    for figure, source in (("fig_robust.pdf", "robustness.json"),):
        fig, data = HERE / "generated" / figure, HERE / "generated" / source
        if fig.exists() and data.exists() and fig.stat().st_mtime < data.stat().st_mtime:
            missing.append(f"{figure} is older than {source}: run make_figures.py")
    abstract = re.search(r"### Abstract\s+(.*?)\n\s*\n", text, flags=re.S)
    if abstract and len(abstract.group(1).strip()) > 1920:
        missing.append(f"abstract is {len(abstract.group(1).strip())} characters, over arXiv's 1,920")
    (HERE / "draft.md").write_text(text + "\n")
    print(f"draft.md: {len(text.split())} words, {len(cited)} distinct citations")
    for name in missing:
        print("missing value:", name)
    for key in uncited:
        print("missing reference:", key)
    for note in notes:
        print("open note:", " ".join(note.split()))
    return 1 if missing or uncited else 0


if __name__ == "__main__":
    sys.exit(main())
