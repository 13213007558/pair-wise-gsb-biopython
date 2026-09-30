# Copyright 2026 by Peter Cock.  All rights reserved.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.
"""Tests for indexed FASTA range access."""

import os
import tempfile
import time
import unittest

from Bio import bgzf
from Bio.Seq import Seq
from Bio.SeqIO import FastaRangeIO


class TestFastaRangeReader(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "sequences.fa")
        self.bgzf_path = os.path.join(self.directory.name, "sequences.fa.bgz")
        self.sequence = "ACGTACGTAC" * 5
        sequence_lines = b"".join(
            self.sequence[i : i + 10].encode("ascii") + b"\n"
            for i in range(0, len(self.sequence), 10)
        )
        self.data = (
            b">chr1 first chromosome\n"
            + sequence_lines
            + b">chr2\n"
            + b"GGG\n"
            + b">empty\n"
        )
        with open(self.path, "wb") as handle:
            handle.write(self.data)

        chunks = [self.data[:21], self.data[21:43], self.data[43:]]
        with open(self.bgzf_path, "wb") as raw:
            for chunk in chunks:
                writer = bgzf.BgzfWriter(fileobj=raw)
                writer._write_block(chunk)
            raw.write(bgzf._bgzf_eof)

        FastaRangeIO.index_fasta(self.path)
        FastaRangeIO.index_fasta(self.bgzf_path)

    def tearDown(self):
        self.directory.cleanup()

    def check_reader(self, path, compressed=False):
        with FastaRangeIO.FastaRangeReader(path) as reader:
            reader.fetch("chr1", 3, 6)
            self.assertGreater(reader.bytes_read, 0)
            self.assertLess(reader.bytes_read, os.path.getsize(path))

            self.assertEqual(reader.fetch("chr1", 0, len(self.sequence)), self.sequence)
            self.assertEqual(reader["chr2"].seq, Seq("GGG"))
            self.assertEqual(reader.fetch("empty", 0, 0), "")

            for start, end in [
                (0, 1),
                (3, 12),
                (9, 31),
                (10, 20),
                (0, 0),
                (7, 7),
                (len(self.sequence), len(self.sequence)),
            ]:
                expected = self.sequence[start:end]
                self.assertEqual(reader.fetch("chr1", start, end), expected)
                self.assertEqual(
                    reader.fetch("chr1", start, end, reverse_complement=True),
                    str(Seq(expected).reverse_complement()),
                )

            self.assertEqual(reader.fetch("chr1"), self.sequence)
            self.assertEqual(
                reader.read("chr1", 2, 8, reverse_complement=True).id,
                "chr1",
            )

            reader.fetch("chr1", 30, 31)
            self.assertLess(reader.bytes_read, os.path.getsize(path))
            reader.fetch("chr1", 12, 12)
            self.assertEqual(reader.bytes_read, 0)

            if compressed:
                with open(path, "rb") as raw:
                    blocks = list(bgzf.BgzfBlocks(raw))
                self.assertEqual(len(blocks), 4)

    def test_plain_file(self):
        self.check_reader(self.path)

    def test_bgzf_file_across_blocks(self):
        with open(self.bgzf_path + ".gzi") as handle:
            gzi_lines = handle.read().splitlines()
        self.assertEqual(gzi_lines[0], "# virtual_offset")
        self.assertGreater(len(gzi_lines), 1)
        self.check_reader(self.bgzf_path, compressed=True)

    def test_crlf_and_short_final_line(self):
        path = os.path.join(self.directory.name, "crlf.fa")
        with open(path, "wb") as handle:
            handle.write(b">chr1\r\nACGT\r\nACGT\r\nAC\r\n")
        FastaRangeIO.index_fasta(path)
        with open(path + ".fai") as handle:
            self.assertEqual(handle.read(), "chr1\t10\t7\t4\t6\n")
        with FastaRangeIO.FastaRangeReader(path) as reader:
            self.assertEqual(reader.fetch("chr1", 0, 10), "ACGTACGTAC")
            self.assertEqual(reader.fetch("chr1", 3, 8), "TACGT")

    def test_final_line_without_newline(self):
        path = os.path.join(self.directory.name, "no_final_newline.fa")
        with open(path, "wb") as handle:
            handle.write(b">chr1\nACGT\nAC")
        FastaRangeIO.index_fasta(path)
        with open(path + ".fai") as handle:
            self.assertEqual(handle.read(), "chr1\t6\t6\t4\t5\n")
        with FastaRangeIO.FastaRangeReader(path) as reader:
            self.assertEqual(reader.fetch("chr1", 0, 6), "ACGTAC")

    def test_external_fai_short_final_line(self):
        path = os.path.join(self.directory.name, "external.fa")
        with open(path, "wb") as handle:
            handle.write(b">chr1\nACGT\nAC\n")
        with open(path + ".fai", "w") as handle:
            handle.write("chr1\t6\t6\t4\t5\n")
        with FastaRangeIO.FastaRangeReader(path) as reader:
            self.assertEqual(reader.fetch("chr1", 0, 6), "ACGTAC")
            self.assertEqual(reader.fetch("chr1", 3, 6), "TAC")

    def test_short_range_does_not_read_whole_record(self):
        whole_record = self.data.index(b">chr2")
        with FastaRangeIO.FastaRangeReader(self.path) as reader:
            reader.fetch("chr1", 3, 6)
            self.assertLess(reader.bytes_read, whole_record)

        with FastaRangeIO.FastaRangeReader(self.bgzf_path) as reader:
            reader.fetch("chr1", 25, 28)
            self.assertLess(reader.bytes_read, os.path.getsize(self.bgzf_path))

    def test_missing_index_is_not_created(self):
        os.remove(self.path + ".fai")
        with self.assertRaises(FastaRangeIO.FastaIndexNotFoundError):
            FastaRangeIO.FastaRangeReader(self.path)
        self.assertFalse(os.path.exists(self.path + ".fai"))

    def test_missing_gzi(self):
        os.remove(self.bgzf_path + ".gzi")
        with self.assertRaises(FastaRangeIO.FastaIndexNotFoundError):
            FastaRangeIO.FastaRangeReader(self.bgzf_path)

    def test_plain_gzip_is_rejected(self):
        import gzip

        path = os.path.join(self.directory.name, "plain.fa.gz")
        with gzip.open(path, "wb") as handle:
            handle.write(b">x\nACGT\n")
        with self.assertRaisesRegex(ValueError, "BGZF"):
            FastaRangeIO.index_fasta(path)

    def test_unknown_and_out_of_range(self):
        with FastaRangeIO.FastaRangeReader(self.path) as reader:
            with self.assertRaises(KeyError):
                reader.fetch("missing", 0, 1)
            with self.assertRaises(FastaRangeIO.FastaRangeError):
                reader.fetch("chr1", 0, len(self.sequence) + 1)
            with self.assertRaises(FastaRangeIO.FastaRangeError):
                reader.fetch("chr1", -1, 1)
            with self.assertRaises(FastaRangeIO.FastaRangeError):
                reader.fetch("chr1", 2, 1)

    def test_duplicate_sequence(self):
        path = os.path.join(self.directory.name, "duplicates.fa")
        with open(path, "wb") as handle:
            handle.write(b">x\nA\n>x\nG\n")
        with self.assertRaises(FastaRangeIO.DuplicateFastaSequenceError):
            FastaRangeIO.index_fasta(path)

    def test_corrupt_fai(self):
        path = self.path + ".fai"
        original = open(path, "rb").read()
        bad_values = [
            b"chr1\t10\t12\n",
            b"chr1\tx\t12\t10\t11\n",
            b"chr1\t10\t12\t0\t11\n",
            original + b"chr1\t40\t70\t10\t11\n",
        ]
        for bad in bad_values:
            with open(path, "wb") as handle:
                handle.write(bad)
            with self.assertRaises(FastaRangeIO.FastaIndexError):
                FastaRangeIO.FastaRangeReader(self.path, check_mtime=False)

    def test_corrupt_gzi(self):
        path = self.bgzf_path + ".gzi"
        for bad in [b"", b"# wrong\n", b"# virtual_offset\nnot-a-number\t2\n"]:
            with open(path, "wb") as handle:
                handle.write(bad)
            with self.assertRaises(FastaRangeIO.CorruptFastaIndexError):
                FastaRangeIO.FastaRangeReader(self.bgzf_path, check_mtime=False)

    def test_stale_indices(self):
        old = time.time_ns() - 2_000_000_000
        os.utime(self.path, ns=(old, old))
        FastaRangeIO.FastaRangeReader(self.path).close()

        os.utime(self.bgzf_path, ns=(old, old))
        FastaRangeIO.FastaRangeReader(self.bgzf_path).close()

        newer = time.time_ns() + 2_000_000_000
        os.utime(self.path, ns=(newer, newer))
        with self.assertRaisesRegex(FastaRangeIO.StaleFastaIndexError, r"\.fai"):
            FastaRangeIO.FastaRangeReader(self.path)

        os.utime(self.bgzf_path, ns=(newer, newer))
        os.utime(self.bgzf_path + ".fai", ns=(newer, newer))
        with self.assertRaisesRegex(FastaRangeIO.StaleFastaIndexError, r"\.gzi"):
            FastaRangeIO.FastaRangeReader(self.bgzf_path)

        with self.assertRaisesRegex(FastaRangeIO.StaleFastaIndexError, r"\.gzi"):
            FastaRangeIO.FastaRangeReader(self.bgzf_path)


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
