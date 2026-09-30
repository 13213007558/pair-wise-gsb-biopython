# Copyright 2026 by the Biopython contributors.  All rights reserved.
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.
"""Random access to regions of FASTA files using samtools style indexes.

Loading a whole chromosome as a single SeqRecord can be wasteful when only
a few hundred bases are needed.  This module provides random access to
intervals of the sequences in a FASTA file, using the same index files as
"samtools faidx":

* a ".fai" text index for plain FASTA files, and
* an additional ".gzi" binary index for BGZF compressed FASTA files.

Indexes are NOT created automatically; call "create_index" explicitly
first, then use "FastaIntervalReader" to fetch intervals::

    from Bio.SeqIO.FastaIndex import create_index, FastaIntervalReader

    create_index("genome.fasta")  # writes genome.fasta.fai
    reader = FastaIntervalReader("genome.fasta")
    record = reader.fetch("chr1", 1000, 1200)  # zero-based, half-open
    record = reader.fetch("chr1", 1000, 1200, reverse_complement=True)
    reader.close()

Coordinates are zero-based half-open, matching Python slicing, so
"fetch(name, start, end)" returns the same sequence as slicing the full
record with "record[start:end]".  The reader works with both LF and CRLF
line endings, and with sequences whose final line is shorter than the
others.  For BGZF files, intervals spanning compressed blocks are handled
transparently.
"""

import os
import struct
from bisect import bisect_right

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

__all__ = [
    "FastaIndexError",
    "DuplicateSequenceNameError",
    "InvalidIndexError",
    "StaleIndexError",
    "IntervalOutOfRangeError",
    "create_index",
    "FastaIntervalReader",
]


class FastaIndexError(Exception):
    """Base class for FASTA index related errors."""


class DuplicateSequenceNameError(FastaIndexError):
    """Raised when a sequence name appears more than once."""


class InvalidIndexError(FastaIndexError):
    """Raised when an index file is corrupt or cannot be parsed."""


class StaleIndexError(FastaIndexError):
    """Raised when an index file is older than the FASTA file."""


class IntervalOutOfRangeError(FastaIndexError, ValueError):
    """Raised when a requested interval falls outside the sequence."""


def _is_gzipped(filename):
    with open(filename, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def _check_bgzf(filename):
    """Return True if the file is BGZF, False if plain, raise if gzip."""
    from Bio import bgzf

    if not _is_gzipped(filename):
        return False
    try:
        with bgzf.BgzfReader(filename, mode="rb"):
            pass
    except ValueError:
        raise FastaIndexError(
            f"{filename!r} is gzipped but not in BGZF (blocked gzip) format; "
            "only BGZF compressed FASTA files support random access"
        ) from None
    return True


def _scan_fasta_lines(handle):
    """Scan binary FASTA lines, returning a list of .fai entries (PRIVATE).

    Each entry is a tuple (name, length, offset, line_bases, line_width).
    """
    entries = []
    names = set()
    name = None
    seq_offset = 0
    length = 0
    line_bases = 0
    line_width = 0
    short_line_seen = False
    pos = 0
    for line in handle:
        if line.startswith(b">"):
            if name is not None:
                entries.append((name, length, seq_offset, line_bases, line_width))
            header = line[1:].split()
            if not header:
                raise FastaIndexError(
                    f"FASTA header without a sequence name at byte offset {pos}"
                )
            name = header[0].decode("ascii", "replace")
            if name in names:
                raise DuplicateSequenceNameError(
                    f"Sequence name {name!r} appears more than once in the FASTA file"
                )
            names.add(name)
            seq_offset = pos + len(line)
            length = 0
            line_bases = 0
            line_width = 0
            short_line_seen = False
        else:
            bases = len(line.rstrip(b"\r\n"))
            if name is None:
                if bases:
                    raise FastaIndexError(
                        "Sequence data found before the first FASTA header"
                    )
            elif bases == 0:
                raise FastaIndexError(
                    f"Blank line in sequence {name!r} at byte offset {pos}; "
                    "blank lines are not supported in indexed FASTA files"
                )
            elif line_bases == 0:
                line_bases = bases
                line_width = len(line)
            else:
                if len(line) != line_width and bases == line_bases:
                    raise FastaIndexError(
                        f"Inconsistent line endings in sequence {name!r}"
                    )
                if bases > line_bases:
                    raise FastaIndexError(
                        f"Sequence line longer than {line_bases} bases in "
                        f"sequence {name!r}"
                    )
                if bases < line_bases:
                    if short_line_seen:
                        raise FastaIndexError(
                            f"More than one short sequence line in sequence {name!r}"
                        )
                    short_line_seen = True
            if name is not None:
                length += bases
        pos += len(line)
    if name is not None:
        entries.append((name, length, seq_offset, line_bases, line_width))
    return entries


def _write_gzi(filename, gzi_path):
    """Build a .gzi index for a BGZF file (PRIVATE)."""
    from Bio import bgzf

    entries = []
    with open(filename, "rb") as handle:
        for start_offset, block_length, data_start, data_len in bgzf.BgzfBlocks(handle):
            if data_start > 0 and data_len > 0:
                entries.append((start_offset, data_start))
    with open(gzi_path, "wb") as handle:
        handle.write(struct.pack("<Q", len(entries)))
        for compressed_offset, uncompressed_offset in entries:
            handle.write(struct.pack("<QQ", compressed_offset, uncompressed_offset))
    return gzi_path


def create_index(filename, index=None, gzi=None):
    """Create a .fai index for a FASTA file, plus .gzi if BGZF compressed.

    Arguments:
     - filename - path to the FASTA file (plain text or BGZF compressed).
     - index - optional path for the .fai index, defaults to the FASTA
       filename plus ".fai".
     - gzi - optional path for the .gzi index (BGZF files only), defaults
       to the FASTA filename plus ".gzi".

    Returns a tuple of the index file paths written (one path for a plain
    FASTA file, two for a BGZF file).

    This is an explicit, separate operation: "FastaIntervalReader" never
    creates or updates index files itself.
    """
    if index is None:
        index = filename + ".fai"
    is_bgzf = _check_bgzf(filename)
    if is_bgzf:
        from Bio import bgzf

        with bgzf.BgzfReader(filename, mode="rb") as handle:
            entries = _scan_fasta_lines(handle)
    else:
        with open(filename, "rb") as handle:
            entries = _scan_fasta_lines(handle)
    with open(index, "w", encoding="utf-8") as handle:
        for name, length, offset, line_bases, line_width in entries:
            handle.write(f"{name}\t{length}\t{offset}\t{line_bases}\t{line_width}\n")
    if is_bgzf:
        if gzi is None:
            gzi = filename + ".gzi"
        return index, _write_gzi(filename, gzi)
    return (index,)


def _parse_fai(index):
    """Parse a .fai file into a dict of name -> (length, offset, line_bases, line_width)."""
    records = {}
    with open(index, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise InvalidIndexError(
                    f"Expected 5 tab-separated fields on line {lineno} of "
                    f"{index!r}, found {len(fields)}"
                )
            name = fields[0]
            if not name:
                raise InvalidIndexError(
                    f"Empty sequence name on line {lineno} of {index!r}"
                )
            try:
                length, offset, line_bases, line_width = (
                    int(field) for field in fields[1:]
                )
            except ValueError:
                raise InvalidIndexError(
                    f"Non-integer value on line {lineno} of {index!r}"
                ) from None
            if length < 0 or offset < 0 or line_bases < 0 or line_width < 0:
                raise InvalidIndexError(
                    f"Negative value on line {lineno} of {index!r}"
                )
            if line_bases == 0:
                if length != 0:
                    raise InvalidIndexError(
                        f"Zero line length but non-zero sequence length on "
                        f"line {lineno} of {index!r}"
                    )
            elif line_width < line_bases:
                raise InvalidIndexError(
                    f"Line width smaller than bases per line on line {lineno} "
                    f"of {index!r}"
                )
            if name in records:
                raise DuplicateSequenceNameError(
                    f"Sequence name {name!r} appears more than once in {index!r}"
                )
            records[name] = (length, offset, line_bases, line_width)
    return records


def _parse_gzi(gzi):
    """Parse a .gzi file into sorted (uncompressed, compressed) offset lists."""
    with open(gzi, "rb") as handle:
        data = handle.read()
    if len(data) < 8:
        raise InvalidIndexError(f"Truncated .gzi index {gzi!r}")
    (count,) = struct.unpack("<Q", data[:8])
    if len(data) != 8 + 16 * count:
        raise InvalidIndexError(
            f"Corrupt .gzi index {gzi!r}: expected {8 + 16 * count} bytes, "
            f"found {len(data)}"
        )
    uncompressed = []
    compressed = []
    prev_u = prev_c = 0
    for i in range(count):
        c, u = struct.unpack_from("<QQ", data, 8 + 16 * i)
        if i and (u <= prev_u or c <= prev_c):
            raise InvalidIndexError(
                f"Corrupt .gzi index {gzi!r}: offsets are not increasing"
            )
        prev_u, prev_c = u, c
        uncompressed.append(u)
        compressed.append(c)
    return uncompressed, compressed


def _check_fresh(index, filename, kind):
    """Raise StaleIndexError if the index is older than the FASTA file."""
    if not os.path.exists(index):
        raise FileNotFoundError(
            f"{kind} index {index!r} not found; call create_index({filename!r}) first"
        )
    if os.path.getmtime(index) < os.path.getmtime(filename):
        raise StaleIndexError(
            f"{kind} index {index!r} is older than {filename!r}; "
            "rebuild it with create_index"
        )


class FastaIntervalReader:
    """Read intervals of sequences from an indexed FASTA file.

    Requires a .fai index (and a .gzi index for BGZF compressed files) as
    produced by "create_index" or "samtools faidx".  Coordinates are
    zero-based half-open, matching Python string slicing.
    """

    def __init__(self, filename, index=None, gzi=None):
        """Open an indexed FASTA file for random access to intervals.

        Arguments:
         - filename - path to the FASTA file (plain text or BGZF).
         - index - optional path of the .fai index, defaults to the FASTA
           filename plus ".fai".
         - gzi - optional path of the .gzi index (BGZF files only),
           defaults to the FASTA filename plus ".gzi".

        Raises FileNotFoundError if an index is missing, StaleIndexError
        if an index is older than the FASTA file, InvalidIndexError if an
        index is corrupt, and DuplicateSequenceNameError if a sequence
        name appears twice in the index.
        """
        if index is None:
            index = filename + ".fai"
        _check_fresh(index, filename, ".fai")
        self._records = _parse_fai(index)
        self._filename = filename
        self._bgzf = _check_bgzf(filename)
        if self._bgzf:
            from Bio import bgzf

            if gzi is None:
                gzi = filename + ".gzi"
            _check_fresh(gzi, filename, ".gzi")
            self._gzi_uncompressed, self._gzi_compressed = _parse_gzi(gzi)
            self._handle = bgzf.BgzfReader(filename, mode="rb")
        else:
            self._handle = open(filename, "rb")

    def __contains__(self, name):
        """Check if a sequence name is present in the index."""
        return name in self._records

    def __len__(self):
        """Return the number of sequences in the index."""
        return len(self._records)

    def __iter__(self):
        """Iterate over the sequence names in the index."""
        return iter(self._records)

    def __getitem__(self, name):
        """Return the full sequence as a SeqRecord."""
        return self.fetch(name)

    def get_length(self, name):
        """Return the length of the named sequence in bases."""
        try:
            return self._records[name][0]
        except KeyError:
            raise KeyError(f"Sequence {name!r} not found in the index") from None

    def _read(self, uncompressed_offset, size):
        if self._bgzf:
            from Bio import bgzf

            i = bisect_right(self._gzi_uncompressed, uncompressed_offset) - 1
            if i >= 0:
                block_start = self._gzi_compressed[i]
                within_block = uncompressed_offset - self._gzi_uncompressed[i]
            else:
                block_start = 0
                within_block = uncompressed_offset
            self._handle.seek(bgzf.make_virtual_offset(block_start, within_block))
        else:
            self._handle.seek(uncompressed_offset)
        return self._handle.read(size)

    def fetch(self, name, start=None, end=None, reverse_complement=False):
        """Fetch an interval of a sequence as a SeqRecord.

        Arguments:
         - name - sequence name (first word of the FASTA header).
         - start - zero-based start coordinate, defaults to 0.
         - end - zero-based end coordinate (exclusive), defaults to the
           sequence length.
         - reverse_complement - if True, return the reverse complement of
           the interval.

        "fetch(name, start, end)" returns the same sequence as slicing
        the full record with "record[start:end]".  An empty interval
        (start equal to end) returns an empty sequence.  Coordinates
        outside the sequence raise IntervalOutOfRangeError, and unknown
        sequence names raise KeyError.
        """
        try:
            length, offset, line_bases, line_width = self._records[name]
        except KeyError:
            raise KeyError(f"Sequence {name!r} not found in the index") from None
        if start is None:
            start = 0
        if end is None:
            end = length
        if not 0 <= start <= end <= length:
            raise IntervalOutOfRangeError(
                f"Interval {start}-{end} is out of range for sequence "
                f"{name!r} of length {length}"
            )
        if start == end:
            seq = Seq("")
        else:
            begin = offset + (start // line_bases) * line_width + start % line_bases
            last = end - 1
            stop = offset + (last // line_bases) * line_width + last % line_bases + 1
            data = self._read(begin, stop - begin)
            seq = Seq(data.replace(b"\n", b"").replace(b"\r", b"").decode("ascii"))
        if reverse_complement:
            seq = seq.reverse_complement()
        return SeqRecord(seq, id=name, name=name, description=f"{name}:{start + 1}-{end}")

    def close(self):
        """Close the underlying file handle."""
        self._handle.close()

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file handle on exit."""
        self.close()

