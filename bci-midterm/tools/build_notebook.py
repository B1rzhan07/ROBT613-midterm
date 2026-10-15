"""Convert a percent-format script (``# %%`` / ``# %% [markdown]``) into .ipynb.

Same source layout as the workshop's ``notebook_src/`` folder, so the notebook
diff stays readable in Git. Usage:

    python tools/build_notebook.py notebook_src/project.py notebooks/project.ipynb
"""
import sys
from pathlib import Path

import nbformat


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                             for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body:
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(src, dst):
    nb = nbformat.v4.new_notebook()
    nb.cells = parse_cells(Path(src).read_text())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, dst)
    print(f"wrote {dst} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
