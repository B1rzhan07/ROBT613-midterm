"""Build and execute the self-contained notebook, saving progress even on failure."""
import argparse, base64, os, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from build_notebook import parse_cells

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='notebooks/project.ipynb');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/a.output;output.parent.mkdir(parents=True,exist_ok=True)
    notebook=nbformat.v4.new_notebook(cells=parse_cells((root/'notebook_src/project.py').read_text()))
    notebook.metadata.kernelspec=dict(name='bci-midterm',display_name='BCI midterm (pinned)',language='python')
    client=NotebookClient(notebook,timeout=1200,kernel_name='bci-midterm',resources={'metadata':{'path':str(root)}})
    def progress(cell,cell_index,**kwargs):
        print(f'Executing cell {cell_index+1}/{len(notebook.cells)}',flush=True)
        nbformat.write(notebook,output)
    client.on_cell_execute=progress
    start=time.perf_counter()
    try:client.execute()
    finally:nbformat.write(notebook,output)
    # Make the submitted notebook portable to Colab and ordinary Python environments.
    notebook.metadata.kernelspec=dict(name='python3',display_name='Python 3',language='python')
    nbformat.write(notebook,output)
    figures=root/os.environ.get('BCI_RESULTS','results')/'figures';figures.mkdir(parents=True,exist_ok=True)
    # Remove only this runner's previously exported plots after a successful execution.
    for old in figures.glob('cell_*.png'): old.unlink()
    for i,c in enumerate(notebook.cells):
        for j,o in enumerate(c.get('outputs',[])):
            if 'image/png' in o.get('data',{}):
                (figures/f'cell_{i:02}_{j}.png').write_bytes(base64.b64decode(o['data']['image/png']))
    print(f'Complete: {output}, wall time {time.perf_counter()-start:.1f} s',flush=True)
if __name__=='__main__':main()
