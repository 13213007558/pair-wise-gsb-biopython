# Copyright 2026 by the Biopython contributors.  All rights reserved.
#
# This file is part of the Biopython distribution and governed by your
# choice of the "Biopython License Agreement" or the "BSD 3-Clause License".
# Please see the LICENSE file that should have been included as part of this
# package.
"""Tests for Bio.SeqIO.FastaIndex (random access to FASTA intervals)."""

import gzip
import os
import random
import shutil
import struct
import tempfile
import unittest

from Bio import bgzf
from Bio.Seq import Seq
from Bio.SeqIO.FastaIndex import (
    DuplicateSequenceNameError,
    FastaIndexError,
    FastaIntervalReader,
    IntervalOutOfRangeError,
    InvalidIndexError,
    StaleIndexError,
    create_index,
)


def make_dna(length, seed):
    """Deterministic pseudo-random DNA sequence of the given length."""
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=length))


def write_fasta(path, records, line_width=60, newline="\n"):
    """Write records (list of (name, sequence)) as a FASTA file."""
    with open(path, "wb") as handle:
        for name, seq in records:
            handle.write(f">{name} some description here{newline}".encode())
            for i in range(0, len(seq), line_width):
                handle.write(seq[i : i + line_width].encode() + newline.encode())


def write_bgzf_fasta(path, records, line_width=60):
    """Write records as a BGZF compressed FASTA file."""
    with bgzf.BgzfWriter(path, mode="wb") as handle:
        for name, seq in records:
            handle.write(f">{name} some description here\n".encode())
            for i in range(0, len(seq), line_width):
                handle.write(seq[i : i + line_width].encode() + b"\n")


class CountingReader:
    """Wrap a binary handle, recording the total number of bytes read."""

    def __init__(self, raw):
        self._raw = raw
        self.bytes_read = 0

    def read(self, size=-1):
        data = self._raw.read(size)
        self.bytes_read += len(data)
        return data

    def seek(self, *args):
        return self._raw.seek(*args)

    def tell(self):
        return self._raw.tell()

    def close(self):
        return self._raw.close()


class BaseFastaIndexTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp(prefix="fasta_index_test_")
        self.addCleanup(shutil.rmtree, self.tempdir)
        self.records = [
            ("chr1", make_dna(5000, seed=1)),  # 5000 = 83 * 60 + 20, short last line
            ("chr2", make_dna(273, seed=2)),
            ("chr3", make_dna(25, seed=3)),  # shorter than one full line
        ]
        self.reference = dict(self.records)

    def path(self, name):
        return os.path.join(self.tempdir, name)

    def check_intervals(self, fasta_path, intervals):
        create_index(fasta_path)
        with FastaIntervalReader(fasta_path) as reader:
            self.assertEqual(set(reader), set(self.reference))
            for name, start, end in self.intervals_for(intervals):
                expected = self.reference[name][start:end]
                with self.subTest(name=name, start=start, end=end):
                    record = reader.fetch(name, start, end)
                    self.assertEqual(str(record.seq), expected)
                    self.assertEqual(len(record.seq), end - start)

    def intervals_for(self, intervals):
        for name in self.reference:
            length = len(self.reference[name])
            for start, end in intervals:
                if 0 <= start <= end <= length:
                    yield name, start, end


class PlainFastaTest(BaseFastaIndexTest):
    """Intervals from a plain LF FASTA file match whole-sequence slices."""

    def test_lf_intervals(self):
        fasta = self.path("test_lf.fasta")
        write_fasta(fasta, self.records, newline="\n")
        intervals = [
            (0, 0),  # empty
            (0, 1),
            (0, 60),  # exactly one line
            (0, 61),  # crosses first line break
            (55, 65),  # crosses a line break
            (59, 120),  # starts at end of a line
            (100, 4833),
            (4950, 5000),  # includes the short last line
            (4999, 5000),  # final base only
            (0, 5000),  # full sequence
            (10, 10),  # empty, non-zero start
            (200, 273),
            (0, 25),
        ]
        self.check_intervals(fasta, intervals)

    def test_crlf_intervals(self):
        """CRLF line endings give the same sequences as LF."""
        fasta = self.path("test_crlf.fasta")
        write_fasta(fasta, self.records, newline="\r\n")
        intervals = [
            (0, 1),
            (55, 65),
            (59, 121),
            (100, 1000),
            (4950, 5000),
            (0, 5000),
            (0, 273),
            (0, 25),
        ]
        self.check_intervals(fasta, intervals)

    def test_reverse_complement(self):
        fasta = self.path("test_rc.fasta")
        write_fasta(fasta, self.records)
        create_index(fasta)
        with FastaIntervalReader(fasta) as reader:
            for name, start, end in [("chr1", 100, 400), ("chr2", 0, 273), ("chr1", 0, 60)]:
                expected = Seq(self.reference[name][start:end]).reverse_complement()
                record = reader.fetch(name, start, end, reverse_complement=True)
                self.assertEqual(str(record.seq), str(expected))

    def test_full_record_access(self):
        fasta = self.path("test_full.fasta")
        write_fasta(fasta, self.records)
        create_index(fasta)
        with FastaIntervalReader(fasta) as reader:
            self.assertEqual(len(reader), 3)
            self.assertIn("chr1", reader)
            self.assertNotIn("chrUn", reader)
            self.assertEqual(reader.get_length("chr1"), 5000)
            self.assertEqual(str(reader["chr2"].seq), self.reference["chr2"])
            self.assertEqual(str(reader.fetch("chr3").seq), self.reference["chr3"])

    def test_short_interval_reads_little(self):
        """Fetching a short interval must not read the whole chromosome."""
        records = [("chrBig", make_dna(200000, seed=7))]
        fasta = self.path("test_reads.fasta")
        write_fasta(fasta, records)
        create_index(fasta)
        reader = FastaIntervalReader(fasta)
        counter = CountingReader(open(fasta, "rb"))
        reader._handle = counter
        try:
            record = reader.fetch("chrBig", 100000, 100200)
            self.assertEqual(str(record.seq), records[0][1][100000:100200])
            # 200 bases over 60-column lines needs ~204 bytes; allow slack,
            # but it must be far below the ~203KB of the full chromosome.
            self.assertLess(counter.bytes_read, 1000)
            self.assertLess(counter.bytes_read, 200000 // 10)
        finally:
            reader.close()


class BgzfFastaTest(BaseFastaIndexTest):
    """BGZF compressed FASTA, using both .fai and .gzi indexes."""

    def setUp(self):
        super().setUp()
        # Long enough to span several 64KB BGZF blocks
        self.records = self.records + [("chrBig", make_dna(200000, seed=11))]
        self.reference = dict(self.records)
        self.fasta = self.path("test.fasta.bgz")
        write_bgzf_fasta(self.fasta, self.records)

    def test_intervals_match_slices(self):
        intervals = [
            (0, 1),
            (55, 65),
            (100, 4833),
            (4950, 5000),
            (0, 5000),
            (0, 273),
            (0, 25),
            (65530, 65540),  # across a 64KB block boundary
            (131060, 131090),  # across the second block boundary
            (199900, 200000),  # tail of the last block
            (0, 200000),  # whole chromosome
        ]
        self.check_intervals(self.fasta, intervals)

    def test_spanning_exact_block_boundary(self):
        """Use the .gzi itself to find a block boundary and straddle it."""
        create_index(self.fasta)
        with open(self.fasta + ".gzi", "rb") as handle:
            (count,) = struct.unpack("<Q", handle.read(8))
            self.assertGreaterEqual(count, 2)
            _, boundary = struct.unpack("<QQ", handle.read(16))
        with FastaIntervalReader(self.fasta) as reader:
            record = reader.fetch("chrBig", boundary - 100, boundary + 100)
            self.assertEqual(
                str(record.seq), self.reference["chrBig"][boundary - 100 : boundary + 100]
            )

    def test_reverse_complement(self):
        create_index(self.fasta)
        with FastaIntervalReader(self.fasta) as reader:
            expected = Seq(self.reference["chrBig"][65500:65600]).reverse_complement()
            record = reader.fetch("chrBig", 65500, 65600, reverse_complement=True)
            self.assertEqual(str(record.seq), str(expected))

    def test_short_interval_reads_few_blocks(self):
        """A short interval must not decompress the whole BGZF file."""
        create_index(self.fasta)
        file_size = os.path.getsize(self.fasta)
        reader = FastaIntervalReader(self.fasta)
        counter = CountingReader(open(self.fasta, "rb"))
        reader._handle._handle = counter  # raw handle inside the BgzfReader
        try:
            record = reader.fetch("chrBig", 150000, 150200)
            self.assertEqual(str(record.seq), self.reference["chrBig"][150000:150200])
            # One or two BGZF blocks, but definitely not the whole file.
            self.assertLess(counter.bytes_read, file_size)
        finally:
            reader.close()

    def test_gzi_written(self):
        paths = create_index(self.fasta)
        self.assertEqual(paths, (self.fasta + ".fai", self.fasta + ".gzi"))
        self.assertTrue(os.path.exists(self.fasta + ".gzi"))

    def test_stale_gzi(self):
        create_index(self.fasta)
        future = os.path.getmtime(self.fasta) + 100
        os.utime(self.fasta + ".fai", (future + 100, future + 100))
        os.utime(self.fasta, (future, future))
        with self.assertRaises(StaleIndexError):
            FastaIntervalReader(self.fasta)

    def test_corrupt_gzi(self):
        create_index(self.fasta)
        with open(self.fasta + ".gzi", "r+b") as handle:
            handle.truncate(13)  # count says more entries than bytes present
        os.utime(self.fasta + ".gzi", None)
        os.utime(self.fasta + ".fai", None)
        with self.assertRaises(InvalidIndexError):
            FastaIntervalReader(self.fasta)


class ErrorHandlingTest(BaseFastaIndexTest):
    def setUp(self):
        super().setUp()
        self.fasta = self.path("test.fasta")
        write_fasta(self.fasta, self.records)

    def test_index_creation_is_explicit(self):
        """No index file is created as a side effect of reading."""
        self.assertFalse(os.path.exists(self.fasta + ".fai"))
        with self.assertRaises(FileNotFoundError):
            FastaIntervalReader(self.fasta)
        self.assertFalse(os.path.exists(self.fasta + ".fai"))
        (index_path,) = create_index(self.fasta)
        self.assertEqual(index_path, self.fasta + ".fai")
        self.assertTrue(os.path.exists(index_path))
        reader = FastaIntervalReader(self.fasta)
        reader.close()

    def test_empty_interval(self):
        create_index(self.fasta)
        with FastaIntervalReader(self.fasta) as reader:
            for start, end in [(0, 0), (100, 100), (5000, 5000)]:
                record = reader.fetch("chr1", start, end)
                self.assertEqual(str(record.seq), "")
                self.assertEqual(len(record.seq), 0)

    def test_interval_out_of_range(self):
        create_index(self.fasta)
        with FastaIntervalReader(self.fasta) as reader:
            for start, end in [(-1, 10), (0, 5001), (50, 40), (5001, 5002), (0, 26)]:
                with self.subTest(start=start, end=end):
                    name = "chr3" if end <= 26 else "chr1"
                    with self.assertRaises(IntervalOutOfRangeError):
                        reader.fetch(name, start, end)
            # Also a ValueError for convenient catching
            with self.assertRaises(ValueError):
                reader.fetch("chr1", 0, 5001)

    def test_unknown_name(self):
        create_index(self.fasta)
        with FastaIntervalReader(self.fasta) as reader:
            with self.assertRaises(KeyError):
                reader.fetch("chrUn", 0, 10)
            with self.assertRaises(KeyError):
                reader.get_length("chrUn")

    def test_duplicate_names_in_fasta(self):
        fasta = self.path("dup.fasta")
        write_fasta(fasta, [("chr1", "ACGT"), ("chr2", "AAAA"), ("chr1", "TTTT")])
        with self.assertRaises(DuplicateSequenceNameError):
            create_index(fasta)

    def test_duplicate_names_in_fai(self):
        create_index(self.fasta)
        with open(self.fasta + ".fai", "a") as handle:
            handle.write("chr1\t4\t100\t4\t5\n")
        os.utime(self.fasta + ".fai", None)
        with self.assertRaises(DuplicateSequenceNameError):
            FastaIntervalReader(self.fasta)

    def test_corrupt_fai(self):
        bad_lines = [
            "chr1\t5000\t30\t60\n",  # too few fields
            "chr1\t5000\t30\t60\t61\textra\n",  # too many fields
            "chr1\tlots\t30\t60\t61\n",  # non-integer
            "chr1\t-5\t30\t60\t61\n",  # negative length
            "chr1\t5000\t30\t60\t59\n",  # width < bases per line
            "chr1\t5000\t30\t0\t0\n",  # zero width but non-zero length
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with open(self.fasta + ".fai", "w") as handle:
                    handle.write(line)
                os.utime(self.fasta + ".fai", None)
                with self.assertRaises(InvalidIndexError):
                    FastaIntervalReader(self.fasta)

    def test_stale_fai(self):
        create_index(self.fasta)
        future = os.path.getmtime(self.fasta + ".fai") + 100
        os.utime(self.fasta, (future, future))  # FASTA now newer than index
        with self.assertRaises(StaleIndexError):
            FastaIntervalReader(self.fasta)

    def test_plain_gzip_rejected(self):
        fasta = self.path("test.fasta.gz")
        with gzip.open(fasta, "wb") as handle:
            handle.write(b">chr1\nACGT\n")
        with self.assertRaises(FastaIndexError):
            create_index(fasta)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
