# Copyright 2026 by Peter Cock.  All rights reserved.
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.
"""Random access to sequence regions in indexed FASTA files.

The indices are the standard ``.fai`` FASTA index and, for BGZF-compressed
FASTA files, the associated ``.gzi`` BGZF index. Creating those files is an
explicit operation, performed by :func:`index_fasta`; opening a FASTA file
with this module does not scan it or silently create an index.

Sequence coordinates are zero-based and half-open. For example, the region
``("chr1", 10, 20)`` contains ten bases, numbered 11 through 20 in the
one-based coordinates commonly used in genome browsers.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import BinaryIO
from typing import Union

from Bio import bgzf
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class FastaIndexError(ValueError):
    """Base class for problems with a FASTA index."""


class FastaIndexNotFoundError(FastaIndexError, FileNotFoundError):
    """A required FASTA or BGZF index file does not exist."""


class DuplicateFastaSequenceError(FastaIndexError):
    """A FASTA index contains the same sequence name more than once."""


class CorruptFastaIndexError(FastaIndexError):
    """A ``.fai`` or ``.gzi`` index cannot be parsed or used."""


class StaleFastaIndexError(FastaIndexError):
    """An index is older than the FASTA file that it describes."""


class FastaRangeError(IndexError, ValueError):
    """A requested sequence region is outside the indexed sequence."""


class _FastaIndexEntry:
    __slots__ = (
        "name",
        "length",
        "offset",
        "line_bases",
        "line_width",
        "short_last_line",
    )

    def __init__(
        self,
        name,
        length,
        offset,
        line_bases,
        line_width,
        short_last_line=False,
    ):
        self.name = name
        self.length = length
        self.offset = offset
        self.line_bases = line_bases
        self.line_width = line_width
        self.short_last_line = short_last_line


class _CountingReader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.handle.read(size)
        self.bytes_read += len(data)
        return data

    def seek(self, offset, whence=os.SEEK_SET):
        return self.handle.seek(offset, whence)

    def tell(self):
        return self.handle.tell()

    def close(self):
        self.handle.close()

    def __getattr__(self, name):
        return getattr(self.handle, name)


class _LazyBgzfReader(bgzf.BgzfReader):
    """BGZF reader which does not decompress a block during initialization."""

    def __init__(self, fileobj):
        self._lazy = True
        self._block_start_offset = None
        self._block_raw_length = None
        self._buffer = b""
        self._within_block_offset = 0
        super().__init__(mode="rb", fileobj=fileobj)
        self._lazy = False

    def _load_block(self, start_offset=None):
        if getattr(self, "_lazy", False) and start_offset == 0:
            return
        super()._load_block(start_offset)


class _BlockTrackingBgzfReader(bgzf.BgzfReader):
    """BGZF reader that records decompressed offsets of compressed blocks."""

    def __init__(self, fileobj):
        self.block_map = [(0, 0)]
        self.uncompressed_position = 0
        super().__init__(mode="rb", fileobj=fileobj)

    def _load_block(self, start_offset=None):
        if start_offset is None and self._block_raw_length:
            previous_data_length = self.uncompressed_position
            previous_buffer_length = len(self._buffer)
            previous_block_end = self._block_start_offset + self._block_raw_length
            super()._load_block(start_offset)
            next_data_start = previous_data_length + previous_buffer_length
            if self._block_raw_length:
                virtual_offset = bgzf.make_virtual_offset(previous_block_end, 0)
                self.block_map.append((next_data_start, virtual_offset))
            self.uncompressed_position = next_data_start
        else:
            super()._load_block(start_offset)


def _is_bgzf(filename):
    with open(filename, "rb") as handle:
        magic = handle.read(4)
    if magic[:2] == b"\x1f\x8b" and magic != bgzf._bgzf_magic:
        raise ValueError(
            "Gzipped FASTA files are not suitable for indexed region access; "
            "please use BGZF (blocked gzip format) instead."
        )
    return magic == bgzf._bgzf_magic


def _line_ending(line):
    if line.endswith(b"\r\n"):
        return line[:-2], 2
    if line.endswith(b"\n"):
        return line[:-1], 1
    if line.endswith(b"\r"):
        raise CorruptFastaIndexError(
            "Bare carriage return line endings are not supported in indexed FASTA"
        )
    return line, 0


def _write_fai(entries, filename):
    with open(filename, "w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            fields = [
                entry.name,
                str(entry.length),
                str(entry.offset),
                str(entry.line_bases),
                str(entry.line_width),
            ]
            handle.write("\t".join(fields) + "\n")


def _write_gzi(block_map, filename):
    with open(filename, "w", encoding="ascii", newline="\n") as handle:
        handle.write("# virtual_offset\n")
        for entry in block_map:
            if len(entry) == 2:
                uncompressed_offset, virtual_offset = entry
            else:
                uncompressed_offset, block_start, within = entry
                virtual_offset = bgzf.make_virtual_offset(block_start, within)
            if uncompressed_offset or virtual_offset:
                handle.write(
                    str(virtual_offset) + "\t" + str(uncompressed_offset) + "\n"
                )


def index_fasta(
    filename: str | os.PathLike[str],
    fai_filename: str | os.PathLike[str] | None = None,
    gzi_filename: str | os.PathLike[str] | None = None,
) -> str:
    """Create a ``.fai`` index, plus a ``.gzi`` index for BGZF FASTA.

    The indices are written to ``filename + '.fai'`` and, when needed,
    ``filename + '.gzi'`` unless explicit filenames are supplied. Existing
    index files are replaced.
    """
    filename = os.fspath(filename)
    compressed = _is_bgzf(filename)
    if fai_filename is None:
        fai_filename = filename + ".fai"
    else:
        fai_filename = os.fspath(fai_filename)
    if compressed and gzi_filename is None:
        gzi_filename = filename + ".gzi"
    elif gzi_filename is not None:
        gzi_filename = os.fspath(gzi_filename)

    raw_handle = open(filename, "rb")
    if compressed:
        handle: Union[_BlockTrackingBgzfReader, BinaryIO] = _BlockTrackingBgzfReader(
            raw_handle
        )
    else:
        handle = raw_handle

    entries: list[_FastaIndexEntry] = []
    uncompressed_offset = 0
    try:
        header = handle.readline()
        while header:
            uncompressed_offset += len(header)
            if not header.startswith(b">"):
                raise CorruptFastaIndexError(
                    "Expected a FASTA header starting with '>', got "
                    + repr(header[:40])
                )
            title = header[1:].rstrip(b"\r\n")
            try:
                name = title.split()[0].decode("utf-8")
            except IndexError:
                name = ""
            except UnicodeDecodeError as err:
                raise CorruptFastaIndexError("Invalid UTF-8 FASTA name") from err
            if "\t" in name:
                raise CorruptFastaIndexError("Tab in FASTA name " + repr(name))
            if any(entry.name == name for entry in entries):
                raise DuplicateFastaSequenceError(
                    "Duplicate FASTA sequence name " + repr(name)
                )

            sequence_offset = uncompressed_offset
            length = 0
            line_bases = 0
            line_width = 0
            line_number = 0
            short_last_line = False
            next_header = b""

            while True:
                line = handle.readline()
                if not line or line.startswith(b">"):
                    if line.startswith(b">"):
                        next_header = line
                    entries.append(
                        _FastaIndexEntry(
                            name,
                            length,
                            sequence_offset,
                            line_bases,
                            line_width,
                            short_last_line,
                        )
                    )
                    break

                bases, ending = _line_ending(line)
                if not bases:
                    raise CorruptFastaIndexError(
                        "Blank line in FASTA sequence " + repr(name)
                    )
                if any(byte in b" \t" for byte in bases):
                    raise CorruptFastaIndexError(
                        "Whitespace in FASTA sequence " + repr(name)
                    )
                current_bases = len(bases)
                current_width = len(line)
                uncompressed_offset += len(line)
                if short_last_line:
                    raise CorruptFastaIndexError(
                        "Only the final FASTA line may be short in " + repr(name)
                    )
                length += current_bases
                line_number += 1
                if line_number == 1:
                    line_bases = current_bases
                    line_width = current_width
                else:
                    if current_bases != line_bases:
                        short_last_line = True
                    elif ending and current_width != line_width:
                        raise CorruptFastaIndexError(
                            "Inconsistent FASTA line widths for " + repr(name)
                        )

            header = next_header
    finally:
        block_map = getattr(handle, "block_map", None)
        handle.close()

    _write_fai(entries, fai_filename)
    if compressed:
        if block_map is None:
            block_map = [(0, 0, 0)]
        _write_gzi(block_map, gzi_filename)
    return fai_filename


def _require_index(path, kind):
    if not os.path.isfile(path):
        raise FastaIndexNotFoundError("Missing " + kind + " index: " + os.fspath(path))
    return path


def _check_not_stale(index_path, fasta_path):
    index_stat = os.stat(index_path)
    fasta_stat = os.stat(fasta_path)
    if index_stat.st_mtime_ns < fasta_stat.st_mtime_ns:
        raise StaleFastaIndexError(
            os.fspath(index_path)
            + " is older than "
            + os.fspath(fasta_path)
            + "; recreate the index"
        )


def _read_fai(path):
    entries = {}
    order = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if not line:
                raise CorruptFastaIndexError(
                    "Empty line in FASTA index " + os.fspath(path)
                )
            fields = line.split("\t")
            if len(fields) != 5:
                raise CorruptFastaIndexError(
                    "FAI line "
                    + str(line_number)
                    + " has "
                    + str(len(fields))
                    + " fields, expected 5"
                )
            name, length, offset, line_bases, line_width = fields
            if not name or "\t" in name:
                raise CorruptFastaIndexError(
                    "Invalid FASTA name in FAI line " + str(line_number)
                )
            try:
                length = int(length)
                offset = int(offset)
                line_bases = int(line_bases)
                line_width = int(line_width)
            except ValueError as err:
                raise CorruptFastaIndexError(
                    "Non-integer FAI field on line " + str(line_number)
                ) from err
            if min(length, offset, line_bases, line_width) < 0:
                raise CorruptFastaIndexError(
                    "Negative FAI field on line " + str(line_number)
                )
            if line_bases == 0 and length:
                raise CorruptFastaIndexError(
                    "Zero line length for non-empty sequence " + repr(name)
                )
            if length and line_width and line_width > line_bases + 2:
                raise CorruptFastaIndexError(
                    "Inconsistent line widths for " + repr(name)
                )
            if name in entries:
                raise DuplicateFastaSequenceError(
                    "Duplicate FASTA sequence name " + repr(name)
                )
            entry = _FastaIndexEntry(
                name,
                length,
                offset,
                line_bases,
                line_width,
                bool(length % line_bases) if line_bases else False,
            )
            entries[name] = entry
            order.append(name)
    return entries, order


def _read_gzi(path):
    with open(path, "rb") as handle:
        data = handle.read()
    if not data:
        raise CorruptFastaIndexError("Empty GZI index " + os.fspath(path))

    if data.startswith(b"#"):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as err:
            raise CorruptFastaIndexError("Invalid text in GZI index") from err
        lines = text.splitlines()
        if lines[0].strip() != "# virtual_offset":
            raise CorruptFastaIndexError("GZI header must be '# virtual_offset'")
        pairs = []
        for line_number, line in enumerate(lines[1:], 2):
            fields = line.split()
            if len(fields) != 2:
                raise CorruptFastaIndexError(
                    "GZI line " + str(line_number) + " does not contain two offsets"
                )
            try:
                left = int(fields[0])
                right = int(fields[1])
            except ValueError as err:
                raise CorruptFastaIndexError(
                    "Non-integer GZI offset on line " + str(line_number)
                ) from err
            pairs.append((left, right))
    block_map = [(0, 0, 0)]
    previous_uncompressed = 0
    previous_virtual = 0
    for left, right in pairs:
        if left < 0 or right < 0:
            raise CorruptFastaIndexError("Negative GZI offset")
        if left == 0 and right == 0:
            continue
        virtual_offset, uncompressed_offset = left, right
        if uncompressed_offset <= previous_uncompressed:
            raise CorruptFastaIndexError("GZI offsets are not increasing")
        if virtual_offset <= previous_virtual:
            raise CorruptFastaIndexError("GZI virtual offsets are not increasing")
        block_start = virtual_offset >> 16
        within = virtual_offset & 0xFFFF
        if block_start == 0 or within != 0:
            raise CorruptFastaIndexError("GZI entry does not start a BGZF block")
        block_map.append((uncompressed_offset, block_start, 0))
        previous_uncompressed = uncompressed_offset
        previous_virtual = virtual_offset
    return block_map


def _validate_gzi_blocks(fasta_path, block_map):
    with open(fasta_path, "rb") as handle:
        for _, compressed_block_start, _ in block_map[1:]:
            handle.seek(compressed_block_start)
            if handle.read(4) != bgzf._bgzf_magic:
                raise CorruptFastaIndexError(
                    "GZI offset does not point to a BGZF block in "
                    + os.fspath(fasta_path)
                )


def _block_for_uncompressed(block_map, offset):
    for uncompressed_block_start, compressed_block_start, within in reversed(block_map):
        if uncompressed_block_start <= offset:
            return bgzf.make_virtual_offset(
                compressed_block_start,
                within + offset - uncompressed_block_start,
            )
    raise CorruptFastaIndexError("No BGZF block contains uncompressed offset")


class FastaRangeReader(Mapping):
    """Read indexed FASTA sequence regions without reading a full record."""

    def __init__(
        self,
        filename: str | os.PathLike[str],
        fai_filename: str | os.PathLike[str] | None = None,
        gzi_filename: str | os.PathLike[str] | None = None,
        *,
        check_mtime: bool = True,
    ) -> None:
        """Open a FASTA file using an existing ``.fai`` and optional ``.gzi``.

        This constructor never creates an index. Call :func:`index_fasta`
        explicitly first. The returned object may be used as a context manager
        and exposes the indexed sequence names as a read-only mapping.
        """
        filename = os.fspath(filename)
        self.filename = filename
        self.compressed = _is_bgzf(self.filename)
        if fai_filename is None:
            fai_filename = self.filename + ".fai"
        else:
            fai_filename = os.fspath(fai_filename)
        if self.compressed and gzi_filename is None:
            gzi_filename = self.filename + ".gzi"
        elif gzi_filename is not None:
            gzi_filename = os.fspath(gzi_filename)

        _require_index(fai_filename, "FASTA (.fai)")
        if check_mtime:
            _check_not_stale(fai_filename, self.filename)
        entries, order = _read_fai(fai_filename)

        if self.compressed:
            _require_index(gzi_filename, "BGZF (.gzi)")
            if check_mtime:
                _check_not_stale(gzi_filename, self.filename)
            self.block_map = _read_gzi(gzi_filename)
            _validate_gzi_blocks(self.filename, self.block_map)
        else:
            self.block_map = None

        self.fai_filename = fai_filename
        self.gzi_filename = gzi_filename
        self._entries = entries
        self._order = tuple(order)
        self._handle = None
        self._raw_handle = None
        self.bytes_read = 0

    def __getitem__(self, name):
        """Return the full indexed sequence as a lightweight :class:`SeqRecord`."""
        try:
            entry = self._entries[name]
        except KeyError:
            raise KeyError(name) from None
        return self.read(name, 0, entry.length)

    def __iter__(self):
        """Iterate over indexed sequence names in index order."""
        return iter(self._order)

    def __len__(self):
        """Return the number of indexed sequences."""
        return len(self._order)

    def _open_handle(self):
        if self._handle is not None:
            return self._handle
        raw_handle = open(self.filename, "rb")
        counting = _CountingReader(raw_handle)
        if self.compressed:
            handle = _LazyBgzfReader(counting)
        else:
            handle = counting
        self._raw_handle = counting
        self.bytes_read = 0
        self._handle = handle
        return handle

    def _entry(self, name):
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError("Unknown FASTA sequence " + repr(name)) from None

    @staticmethod
    def _start_offset(entry, position):
        full_bases = entry.length // entry.line_bases * entry.line_bases
        if entry.short_last_line and position > full_bases:
            full_lines = entry.length // entry.line_bases
            return entry.offset + full_lines * entry.line_width + position - full_bases
        line, remainder = divmod(position, entry.line_bases)
        return entry.offset + line * entry.line_width + remainder

    @staticmethod
    def _end_offset(entry, position):
        if position == 0:
            return entry.offset
        full_bases = entry.length // entry.line_bases * entry.line_bases
        if entry.short_last_line and position > full_bases:
            full_lines = entry.length // entry.line_bases
            return entry.offset + full_lines * entry.line_width + position - full_bases
        line, remainder = divmod(position - 1, entry.line_bases)
        return entry.offset + line * entry.line_width + remainder + 1

    def _read_range(self, entry, start, end):
        if start == end:
            return b""
        start_offset = self._start_offset(entry, start)
        end_offset = self._end_offset(entry, end)
        handle = self._open_handle()
        self._raw_handle.bytes_read = 0
        if self.compressed:
            handle.seek(_block_for_uncompressed(self.block_map, start_offset))
            size = end_offset - start_offset
        else:
            handle.seek(start_offset)
            size = end_offset - start_offset
        data = handle.read(size)
        if len(data) != size:
            raise CorruptFastaIndexError(
                "FASTA file is shorter than the index indicates"
            )
        return data

    @staticmethod
    def _clean(data):
        return data.translate(None, b"\r\n")

    def fetch(self, name, start=None, end=None, *, reverse_complement=False):
        """Return a sequence region as a Python string.

        Coordinates are zero-based and half-open. ``start`` defaults to zero
        and ``end`` defaults to the indexed sequence length. Empty regions
        return an empty string.
        """
        entry = self._entry(name)
        if start is None:
            start = 0
        if end is None:
            end = entry.length
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("FASTA region start and end must be integers")
        if start < 0 or end < 0:
            raise FastaRangeError("Negative FASTA coordinate")
        if end > entry.length:
            raise FastaRangeError(
                "End "
                + str(end)
                + " is beyond length "
                + str(entry.length)
                + " for "
                + repr(name)
            )
        if start > end:
            raise FastaRangeError("Start " + str(start) + " is after end " + str(end))

        self.bytes_read = 0
        if start == end and self._raw_handle is not None:
            self._raw_handle.bytes_read = 0
        data = self._clean(self._read_range(entry, start, end))
        if len(data) != end - start:
            raise CorruptFastaIndexError(
                "Indexed line geometry does not match the FASTA file"
            )
        sequence = data.decode("ascii")
        if reverse_complement:
            sequence = str(Seq(sequence).reverse_complement())
        self.bytes_read = self._raw_handle.bytes_read if self._raw_handle else 0
        return sequence

    def read(self, name, start=None, end=None, *, reverse_complement=False):
        """Return a sequence region as a lightweight :class:`SeqRecord`."""
        sequence = self.fetch(
            name,
            start,
            end,
            reverse_complement=reverse_complement,
        )
        return SeqRecord(Seq(sequence), id=name, name=name, description="")

    def close(self):
        """Close the underlying FASTA file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._raw_handle = None
            self.bytes_read = 0

    def __enter__(self):
        """Return the reader."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the reader on context-manager exit."""
        self.close()


def open_fasta(
    filename: str | os.PathLike[str],
    fai_filename: str | os.PathLike[str] | None = None,
    gzi_filename: str | os.PathLike[str] | None = None,
    *,
    check_mtime: bool = True,
) -> FastaRangeReader:
    """Open an indexed FASTA file for zero-based region retrieval."""
    return FastaRangeReader(
        filename,
        fai_filename=fai_filename,
        gzi_filename=gzi_filename,
        check_mtime=check_mtime,
    )


FastaRandomAccess = FastaRangeReader
FastaIndexedReader = FastaRangeReader
create_fasta_index = index_fasta
build_fasta_index = index_fasta
