"""Fetch required table CADs from PointFlow's original PC15k ZIP using HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import struct
import time
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://drive.usercontent.google.com/download?id=1sw9gdk_igiyyt7MqALyxZhRrtPvAn0sX&export=download&confirm=t'


def fetch_range(start, end):
    for attempt in range(4):
        try:
            request = urllib.request.Request(URL + f'&range={start}-{end}', headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                content_range = response.headers.get('Content-Range', '')
                if response.status != 206 or not content_range.startswith(f'bytes {start}-'):
                    raise RuntimeError(f'Server did not honor requested range: {content_range}')
                payload = response.read()
                if len(payload) != end - start + 1:
                    raise RuntimeError('Incomplete HTTP range')
                return payload, int(content_range.split('/')[-1])
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
        _, self.length = fetch_range(0, 0)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b''
        payload, _ = fetch_range(self.position, self.position + size - 1)
        self.position += size
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/ShapeNetCore.v2.PC15k/04379243')
    args = parser.parse_args()
    ids = {Path(line.split()[0]).stem for line in
           (ROOT / 'configs/04379243_scan2cad_rot_sym_label.txt').read_text().splitlines()}
    archive = RemoteZip()
    with zipfile.ZipFile(archive) as zipped:
        members = [entry for entry in zipped.infolist() if '/04379243/' in entry.filename
                   and Path(entry.filename).stem in ids and entry.filename.endswith('.npy')]
    if {Path(entry.filename).stem for entry in members} != ids:
        raise RuntimeError('Original archive is missing required table CAD IDs')

    def extract(entry):
        split = Path(entry.filename).parent.name
        if split not in ['train', 'test', 'val']:
            raise RuntimeError('Unexpected dataset split')
        destination = args.output / split / Path(entry.filename).name
        if destination.exists() and zlib.crc32(destination.read_bytes()) == entry.CRC:
            return
        payload, _ = fetch_range(entry.header_offset,
            min(archive.length - 1, entry.header_offset + entry.compress_size + 4095))
        fields = struct.unpack('<4s5H3I2H', payload[:30])
        if fields[0] != b'PK\x03\x04':
            raise RuntimeError('Invalid ZIP local header')
        offset = 30 + fields[-2] + fields[-1]
        compressed = payload[offset:offset + entry.compress_size]
        data = zlib.decompress(compressed, -15) if entry.compress_type == zipfile.ZIP_DEFLATED else compressed
        if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
            raise RuntimeError(f'CRC/size mismatch: {entry.filename}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.npy.part')
        temporary.write_bytes(data)
        temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, _ in enumerate(pool.map(extract, members), 1):
            if index % 50 == 0 or index == len(members):
                print(f'Validated table CADs {index}/{len(members)}', flush=True)
    print(f'Table CADs ready: {args.output}', flush=True)


if __name__ == '__main__':
    main()
