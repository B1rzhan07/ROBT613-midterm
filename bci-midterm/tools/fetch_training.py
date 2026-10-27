"""Fetch unmodified training MAT members from public Zenodo ZIPs using HTTP ranges.

This is archive retrieval only. MOABB's BI2013a reader still creates all MNE objects.
ZIP CRC is verified by RemoteZip; SHA-256 records the extracted local bytes.
"""
import hashlib,json,time,io,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from remotezip import RemoteZip, RemoteFetcher, PartialBuffer
from requests import HTTPError
from email.utils import parsedate_to_datetime

_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0


class ParallelRangeFetcher(RemoteFetcher):
    """Bound streams, pace requests globally, and respect server retry instructions."""
    def _request(self, kwargs):
        global _LAST_REQUEST
        for attempt in range(6):
            with _REQUEST_LOCK:
                wait = 1.0 - (time.monotonic() - _LAST_REQUEST)
                if wait > 0: time.sleep(wait)
                _LAST_REQUEST = time.monotonic()
            try:
                return super()._request(kwargs)
            except HTTPError as error:
                response = error.response
                if response is None or response.status_code not in (429, 503) or attempt == 5:
                    raise
                retry_after = response.headers.get('Retry-After', '')
                try:
                    delay = max(1., float(retry_after))
                except ValueError:
                    try:
                        delay = max(1., parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (ValueError, TypeError):
                        delay = min(120., 15. * 2**attempt)
                response.close()
                with _REQUEST_LOCK:
                    _LAST_REQUEST = max(_LAST_REQUEST, time.monotonic() + delay)
                print(f'Zenodo requested a pause; retrying in {delay:.0f} s', flush=True)
                time.sleep(delay)

    def fetch(self, data_range, stream=False):
        first, last = data_range
        if first < 0 or last is None or last - first < 8 * 1024**2:
            return super().fetch(data_range, stream)
        blocks = [(a, min(a + 4 * 1024**2 - 1, last)) for a in range(first, last + 1, 4 * 1024**2)]
        def block(bounds):
            for attempt in range(3):
                try:
                    part = super(ParallelRangeFetcher, self).fetch(bounds, stream=False)
                    try:
                        data = part.read()
                        if part.tell() != bounds[1] + 1 or len(data) != bounds[1] - bounds[0] + 1:
                            raise IOError('Incomplete or incorrect HTTP byte range')
                        return data
                    finally:
                        part.close()
                except Exception:
                    if attempt == 2: raise
                    time.sleep(attempt + 1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            data = b''.join(pool.map(block, blocks))
        return PartialBuffer(io.BytesIO(data), first, len(data), stream=False)


def existing_directory(root,subject):
    root=Path(root)
    for directory in [root/'BI2013a-training-only'/f'subject_{subject:02}',
                      root/'MNE-braininvaders2013-data/zenodo/2669187'/f'subject_{subject:02}']:
        meta_path=directory/'meta.yml'
        if not meta_path.exists():continue
        meta=yaml.safe_load(meta_path.read_text())
        names=[r['filename'] for r in meta['runs'] if r['type']=='training']
        paths=[directory/'Session1'/n.replace('.gdf','.mat') for n in names]
        if len(paths)==2 and all(p.exists() for p in paths):return directory,paths
    return None


def fetch_subject(root,subject):
    if not 8<=subject<=24:raise ValueError('This helper supports the declared single-session subset 8-24 only')
    found=existing_directory(root,subject)
    if found:return [str(p) for p in found[1]]
    directory=Path(root)/'BI2013a-training-only'/f'subject_{subject:02}'
    directory.mkdir(parents=True,exist_ok=True)
    url=f'https://zenodo.org/records/2669187/files/subject{subject:02}.zip?download=1'
    prefix=f'subject{subject:02}'
    log=[]
    for attempt in range(3):
        try:
            with RemoteZip(url,timeout=45,fetcher=ParallelRangeFetcher) as archive:
                meta_bytes=archive.read(f'{prefix}/meta.yml')
                meta=yaml.safe_load(meta_bytes)
                (directory/'meta.yml').write_bytes(meta_bytes)
                names=[r['filename'].replace('.gdf','.mat') for r in meta['runs'] if r['type']=='training']
                for name in names:
                    dest=directory/'Session1'/name
                    if dest.exists():continue
                    start=time.perf_counter()
                    data=archive.read(f'{prefix}/Session1/{name}')
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    temporary=dest.with_suffix('.partial');temporary.write_bytes(data);temporary.replace(dest)
                    log.append(dict(member=f'{prefix}/Session1/{name}',bytes=len(data),
                                    sha256=hashlib.sha256(data).hexdigest(),seconds=time.perf_counter()-start))
            (directory/'download_manifest.json').write_text(json.dumps(dict(url=url,files=log),indent=2))
            return [str(directory/'Session1'/n) for n in names]
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))


def prepare_subjects(root,subjects,workers=4):
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(fetch_subject,root,s):s for s in subjects}
        for future in as_completed(jobs):
            paths=future.result()
            print(f'Training files ready: subject {jobs[future]} ({len(paths)} runs)',flush=True)
    print(f'Data preparation: {time.perf_counter()-start:.1f} s',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='./mne_data');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();prepare_subjects(a.root,range(8,25),a.workers)
