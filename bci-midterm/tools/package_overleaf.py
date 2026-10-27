"""Package the current report source and figures as a standalone Overleaf project."""
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
OUTPUT = ROOT / 'output' / 'overleaf-report.zip'

README = """# Редактирование отчёта в Overleaf

Загрузите overleaf-report.zip как новый проект в Overleaf.
Основной файл: main.tex. Компилятор: XeLaTeX (соответствует проверенной сборке).

В архиве находится текущая версия отчёта из report/main.tex и все девять
рисунков из report/images/. Пакет IEEEtran и остальные стандартные пакеты
доступны в TeX Live / Overleaf; отдельная установка Python не нужна.

Что редактировать:
- main.tex: заголовок, авторы, аннотация, разделы, таблицы, подписи и литература.
- images/: рисунки; при замене сохраняйте имена или меняйте includegraphics.
- Ссылки на литературу находятся в thebibliography в конце main.tex;
  отдельный .bib-файл не требуется.

Отчёт использует IEEEtran, две колонки, шесть страниц основного текста
и отдельную страницу литературы. После правок проверьте переносы и число страниц.
Числа в таблицах перенесены из результатов выполненного notebook; их изменение
в LaTeX не перезапускает анализ. Правки в Overleaf не синхронизируются с GitHub
или notebook автоматически. tools/build_report.py в репозитории заново генерирует
report/main.tex и может перезаписать ручные правки — сохраните экспорт из Overleaf.

Этот ZIP содержит только исходники для Overleaf, без результатов EEG и PDF.
"""


def main():
    source = (REPORT / 'main.tex').read_text()
    names = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source)
    assets = [REPORT / name for name in sorted(set(names))]
    assert assets and all(p.is_file() for p in assets)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(REPORT / 'main.tex', 'main.tex')
        for asset in assets:
            archive.write(asset, asset.relative_to(REPORT))
        archive.writestr('README.md', README)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert archive.read('main.tex') == (REPORT / 'main.tex').read_bytes()
        assert len(archive.namelist()) == 2 + len(assets)
    print(f'Created {OUTPUT}: main.tex, {len(assets)} figures, README.md')


if __name__ == '__main__':
    main()
