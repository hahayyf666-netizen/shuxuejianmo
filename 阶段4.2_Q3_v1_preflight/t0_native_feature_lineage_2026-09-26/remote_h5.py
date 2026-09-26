"""Small read-only HTTP range file for inspecting pinned public HDF5 CSD assets.

The file object verifies every 206 Content-Range response. It does not assert a
whole-file SHA: a pinned repository commit and response ETag are recorded instead.
"""

from __future__ import annotations

import io
import re
from collections import OrderedDict

import requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, *, block_size: int = 1024 * 1024, max_blocks: int = 64):
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.session = requests.Session()
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.requests_made = 0
        self.bytes_transferred = 0
        first = self.session.get(url, headers={"Range": "bytes=0-15"}, timeout=45)
        first.raise_for_status()
        size_match = re.fullmatch(r"bytes 0-15/(\d+)", first.headers.get("Content-Range", ""))
        if first.status_code != 206 or not size_match or first.content[:8] != b"\x89HDF\r\n\x1a\n":
            raise ValueError("remote asset is not a byte-range-readable HDF5 file")
        self.size = int(size_match.group(1))
        self.etag = first.headers.get("ETag") or first.headers.get("X-Linked-Etag")
        self.requests_made += 1
        self.bytes_transferred += len(first.content)

    def _block(self, number: int) -> bytes:
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        start = number * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        if response.status_code != 206 or content_range != f"bytes {start}-{end}/{self.size}":
            raise IOError(f"invalid Content-Range: {content_range!r}")
        response_etag = response.headers.get("ETag") or response.headers.get("X-Linked-Etag")
        if self.etag and response_etag and response_etag != self.etag:
            raise IOError("remote candidate ETag changed during range read")
        if len(response.content) != end - start + 1:
            raise IOError("remote block length differs from advertised range")
        self.requests_made += 1
        self.bytes_transferred += len(response.content)
        self.cache[number] = response.content
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return response.content

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        else:
            raise ValueError("invalid whence")
        if self.position < 0:
            raise ValueError("negative seek")
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        out = bytearray()
        while size:
            block_number, offset = divmod(self.position, self.block_size)
            block = self._block(block_number)
            chunk = block[offset:offset + size]
            out.extend(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return bytes(out)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
        super().close()
