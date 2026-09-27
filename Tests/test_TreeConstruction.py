# Copyright (C) 2013 by Yanbo Ye (yeyanbo289@gmail.com)
# This code is part of the Biopython distribution and governed by its
# license. Please see the LICENSE file that should have been included
# as part of this package.

"""Unit tests for the Bio.Phylo.TreeConstruction module."""

import os
import random
import tempfile
import unittest
from io import StringIO
from itertools import combinations, product

from Bio import Align
from Bio import AlignIO
from Bio import Phylo
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
from Bio.Phylo import BaseTree
from Bio.Phylo import Consensus
from Bio.Phylo import TreeConstruction
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import _Matrix
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.TreeConstruction import DistanceMatrix
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import NNITreeSearcher
from Bio.Phylo.TreeConstruction import ParsimonyScorer
from Bio.Phylo.TreeConstruction import ParsimonyTreeConstructor

temp_dir = tempfile.mkdtemp()


class DistanceMatrixTest(unittest.TestCase):
    """Test for DistanceMatrix construction and manipulation."""

    def setUp(self):
        self.names = ["Alpha", "Beta", "Gamma", "Delta"]
        self.matrix = [[0], [1, 0], [2, 3, 0], [4, 5, 6, 0]]

    def test_good_construction(self):
        dm = DistanceMatrix(self.names, self.matrix)
        self.assertIsInstance(dm, TreeConstruction.DistanceMatrix)
        self.assertEqual(dm.names[0], "Alpha")
        self.assertEqual(dm.matrix[2][1], 3)
        self.assertEqual(len(dm), 4)
        self.assertEqual(
            repr(dm),
            "DistanceMatrix(names=['Alpha', 'Beta', 'Gamma', 'Delta'], "
            "matrix=[[0], [1, 0], [2, 3, 0], [4, 5, 6, 0]])",
        )

    def test_bad_construction(self):
        self.assertRaises(
            TypeError,
            DistanceMatrix,
            ["Alpha", 100, "Gamma", "Delta"],
            [[0], [0.1, 0], [0.2, 0.3, 0], [0.4, 0.5, 0.6, 0]],
        )
        self.assertRaises(
            TypeError,
            DistanceMatrix,
            ["Alpha", "Beta", "Gamma", "Delta"],
            [[0], ["a"], [0.2, 0.3], [0.4, 0.5, 0.6]],
        )
        self.assertRaises(
            ValueError,
            DistanceMatrix,
            ["Alpha", "Alpha", "Gamma", "Delta"],
            [[0], [0.1], [0.2, 0.3], [0.4, 0.5, 0.6]],
        )
        self.assertRaises(
            ValueError,
            DistanceMatrix,
            ["Alpha", "Beta", "Gamma", "Delta"],
            [[0], [0.2, 0], [0.4, 0.5, 0.6]],
        )
        self.assertRaises(
            ValueError,
            DistanceMatrix,
            ["Alpha", "Beta", "Gamma", "Delta"],
            [[0], [0.1], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]],
        )

    def test_good_manipulation(self):
        dm = DistanceMatrix(self.names, self.matrix)
        # getitem
        self.assertEqual(dm[1], [1, 0, 3, 5])
        self.assertEqual(dm[2, 1], 3)
        self.assertEqual(dm[2][1], 3)
        self.assertEqual(dm[1, 2], 3)
        self.assertEqual(dm[1][2], 3)
        self.assertEqual(dm["Alpha"], [0, 1, 2, 4])
        self.assertEqual(dm["Gamma", "Delta"], 6)
        # setitem
        dm["Alpha"] = [0, 10, 20, 40]
        self.assertEqual(dm["Alpha"], [0, 10, 20, 40])
        # delitem insert item
        del dm[1]
        self.assertEqual(dm.names, ["Alpha", "Gamma", "Delta"])
        self.assertEqual(dm.matrix, [[0], [20, 0], [40, 6, 0]])
        dm.insert("Beta", [1, 0, 3, 5], 1)
        self.assertEqual(dm.names, self.names)
        self.assertEqual(dm.matrix, [[0], [1, 0], [20, 3, 0], [40, 5, 6, 0]])
        del dm["Alpha"]
        self.assertEqual(dm.names, ["Beta", "Gamma", "Delta"])
        self.assertEqual(dm.matrix, [[0], [3, 0], [5, 6, 0]])
        dm.insert("Alpha", [1, 2, 4, 0])
        self.assertEqual(dm.names, ["Beta", "Gamma", "Delta", "Alpha"])
        self.assertEqual(dm.matrix, [[0], [3, 0], [5, 6, 0], [1, 2, 4, 0]])

    def test_bad_manipulation(self):
        dm = DistanceMatrix(self.names, self.matrix)
        # getitem
        self.assertRaises(ValueError, dm.__getitem__, "A")
        self.assertRaises(ValueError, dm.__getitem__, ("Alpha", "A"))
        self.assertRaises(TypeError, dm.__getitem__, (1, "A"))
        self.assertRaises(TypeError, dm.__getitem__, (1, 1.2))
        self.assertRaises(IndexError, dm.__getitem__, 6)
        self.assertRaises(IndexError, dm.__getitem__, (10, 10))
        # setitem: item or index test
        self.assertRaises(ValueError, dm.__setitem__, "A", [1, 3, 4])
        self.assertRaises(ValueError, dm.__setitem__, ("Alpha", "A"), 4)
        self.assertRaises(TypeError, dm.__setitem__, (1, "A"), 3)
        self.assertRaises(TypeError, dm.__setitem__, (1, 1.2), 2)
        self.assertRaises(IndexError, dm.__setitem__, 6, [1, 3, 4])
        self.assertRaises(IndexError, dm.__setitem__, (10, 10), 1)
        # setitem: value test
        self.assertRaises(ValueError, dm.__setitem__, 0, [1, 2])
        self.assertRaises(TypeError, dm.__setitem__, ("Alpha", "Beta"), "a")
        self.assertRaises(TypeError, dm.__setitem__, "Alpha", ["a", "b", "c"])

    def test_format_phylip(self):
        dm = DistanceMatrix(self.names, self.matrix)
        handle = StringIO()
        dm.format_phylip(handle)
        lines = handle.getvalue().splitlines()
        self.assertEqual(len(lines), len(dm) + 1)
        self.assertTrue(lines[0].endswith(str(len(dm))))
        for name, line in zip(self.names, lines[1:]):
            self.assertTrue(line.startswith(name))


class DistanceCalculatorTest(unittest.TestCase):
    """Test DistanceCalculator."""

    def test_known_matrices_msa(self):
        msa = AlignIO.read("TreeConstruction/msa.phy", "phylip")

        calculator = DistanceCalculator("identity")
        dm = calculator.get_distance(msa)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 10 / 13)

        calculator = DistanceCalculator("blastn")
        dm = calculator.get_distance(msa)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 38 / 65)

        calculator = DistanceCalculator("trans")
        dm = calculator.get_distance(msa)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 54 / 65)

        calculator = DistanceCalculator("blosum62")
        dm = calculator.get_distance(msa)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 53 / 84)

    def test_known_matrices(self):
        aln = Align.read("TreeConstruction/msa.phy", "phylip")

        calculator = DistanceCalculator("identity")
        dm = calculator.get_distance(aln)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 10 / 13)

        calculator = DistanceCalculator("blastn")
        dm = calculator.get_distance(aln)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 38 / 65)

        calculator = DistanceCalculator("trans")
        dm = calculator.get_distance(aln)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 54 / 65)

        calculator = DistanceCalculator("blosum62")
        dm = calculator.get_distance(aln)
        self.assertEqual(dm["Alpha", "Beta"], 1 - 53 / 84)

    def test_nonmatching_seqs_msa(self):
        aln = AlignIO.read(StringIO(">Alpha\nA-A--\n>Gamma\n-Y-Y-"), "fasta")
        # With a proper scoring matrix -- no matches
        dmat = DistanceCalculator("blosum62").get_distance(aln)
        self.assertEqual(dmat["Alpha", "Alpha"], 0.0)
        self.assertEqual(dmat["Alpha", "Gamma"], 1.0)
        # Comparing characters only -- 4 misses, 1 match
        dmat = DistanceCalculator().get_distance(aln)
        self.assertEqual(dmat["Alpha", "Alpha"], 0.0)
        self.assertAlmostEqual(dmat["Alpha", "Gamma"], 4.0 / 5.0)

    def test_nonmatching_seqs(self):
        aln = Align.read(
            StringIO(">Alpha\nA-A--\n>Beta\nXXXXX\n>Gamma\n-Y-Y-"), "fasta"
        )
        # With a proper scoring matrix -- no matches
        dmat = DistanceCalculator("blosum62").get_distance(aln)
        self.assertEqual(dmat["Alpha", "Alpha"], 0.0)
        self.assertEqual(dmat["Alpha", "Gamma"], 1.0)
        # Comparing characters only -- 4 misses, 1 match
        dmat = DistanceCalculator().get_distance(aln)
        self.assertEqual(dmat["Alpha", "Alpha"], 0.0)
        self.assertAlmostEqual(dmat["Alpha", "Gamma"], 4.0 / 5.0)


class DistanceTreeConstructorTest(unittest.TestCase):
    """Test DistanceTreeConstructor."""

    def setUp(self):
        self.msa = AlignIO.read("TreeConstruction/msa.phy", "phylip")
        calculator = DistanceCalculator("blosum62")
        self.dm_msa = calculator.get_distance(self.msa)
        self.constructor_msa = DistanceTreeConstructor(calculator)
        self.alignment = Align.read("TreeConstruction/msa.phy", "phylip")
        calculator = DistanceCalculator("blosum62")
        self.dm = calculator.get_distance(self.alignment)
        self.constructor = DistanceTreeConstructor(calculator)

    def test_upgma_msa(self):
        tree = self.constructor.upgma(self.dm_msa)
        self.assertIsInstance(tree, BaseTree.Tree)
        # tree_file = StringIO()
        # Phylo.write(tree, tree_file, 'newick')
        ref_tree = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        self.assertTrue(Consensus._equal_topology(tree, ref_tree))
        # ref_tree.close()

    def test_upgma(self):
        tree = self.constructor.upgma(self.dm)
        self.assertIsInstance(tree, BaseTree.Tree)
        # tree_file = StringIO()
        # Phylo.write(tree, tree_file, 'newick')
        ref_tree = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        self.assertTrue(Consensus._equal_topology(tree, ref_tree))
        # check for equal distance of all terminal nodes from the root
        ref_tree.root_at_midpoint()
        for len1, len2 in combinations(
            [depth for node, depth in ref_tree.depths().items() if node.is_terminal()],
            2,
        ):
            self.assertAlmostEqual(len1, len2)

    def test_nj_msa(self):
        tree = self.constructor.nj(self.dm_msa)
        self.assertIsInstance(tree, BaseTree.Tree)
        # tree_file = StringIO()
        # Phylo.write(tree, tree_file, 'newick')
        ref_tree = Phylo.read("./TreeConstruction/nj.tre", "newick")
        self.assertTrue(Consensus._equal_topology(tree, ref_tree))
        # ref_tree.close()

        # create a matrix of length 2
        calculator = DistanceCalculator("blosum62")
        self.min_dm = calculator.get_distance(self.msa)
        for i in range(len(self.min_dm) - 2):
            del self.min_dm[len(self.min_dm) - 1]

        min_tree = self.constructor.nj(self.min_dm)
        self.assertIsInstance(min_tree, BaseTree.Tree)

        ref_min_tree = Phylo.read("./TreeConstruction/nj_min.tre", "newick")
        self.assertTrue(Consensus._equal_topology(min_tree, ref_min_tree))

    def test_nj(self):
        tree = self.constructor.nj(self.dm)
        self.assertIsInstance(tree, BaseTree.Tree)
        # tree_file = StringIO()
        # Phylo.write(tree, tree_file, 'newick')
        ref_tree = Phylo.read("./TreeConstruction/nj.tre", "newick")
        self.assertTrue(Consensus._equal_topology(tree, ref_tree))
        # ref_tree.close()

        # create a matrix of length 2
        calculator = DistanceCalculator("blosum62")
        self.min_dm = calculator.get_distance(self.msa)
        for i in range(len(self.min_dm) - 2):
            del self.min_dm[len(self.min_dm) - 1]

        min_tree = self.constructor.nj(self.min_dm)
        self.assertIsInstance(min_tree, BaseTree.Tree)

        ref_min_tree = Phylo.read("./TreeConstruction/nj_min.tre", "newick")
        self.assertTrue(Consensus._equal_topology(min_tree, ref_min_tree))

    def test_built_tree_msa(self):
        tree = self.constructor.build_tree(self.msa)
        self.assertIsInstance(tree, BaseTree.Tree)
        # tree_file = StringIO()
        # Phylo.write(tree, tree_file, 'newick')
        ref_tree = Phylo.read("./TreeConstruction/nj.tre", "newick")
        self.assertTrue(Consensus._equal_topology(tree, ref_tree))
        # ref_tree.close()

    def test_built_tree(self):
        tree = self.constructor.build_tree(self.alignment)
        self.assertIsInstance(tree, BaseTree.Tree)
        # tree_file = StringIO()
        # Phylo.write(tree, tree_file, 'newick')
        ref_tree = Phylo.read("./TreeConstruction/nj.tre", "newick")
        self.assertTrue(Consensus._equal_topology(tree, ref_tree))
        # ref_tree.close()


class ParsimonyScorerTest(unittest.TestCase):
    """Test ParsimonyScorer."""

    def test_get_score_msa(self):
        aln = AlignIO.read("TreeConstruction/msa.phy", "phylip")
        tree = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        scorer = ParsimonyScorer()
        score = scorer.get_score(tree, aln)
        self.assertEqual(score, 2 + 1 + 2 + 2 + 1 + 1 + 1 + 3)

        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2.5, 0], [2.5, 1, 0], [1, 2.5, 2.5, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        score = scorer.get_score(tree, aln)
        self.assertEqual(score, 3.5 + 2.5 + 3.5 + 3.5 + 2.5 + 1 + 2.5 + 4.5)

        alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "1",
            "2",
            "T",
            "V",
            "W",
            "Y",
            "*",
            "-",
        ]
        step_matrix = [
            [0],
            [2, 0],
            [1, 2, 0],
            [1, 2, 1, 0],
            [2, 1, 2, 2, 0],
            [1, 1, 1, 1, 2, 0],
            [2, 2, 1, 2, 2, 2, 0],
            [2, 2, 2, 2, 1, 2, 2, 0],
            [2, 2, 2, 1, 2, 2, 2, 1, 0],
            [2, 2, 2, 2, 1, 2, 1, 1, 2, 0],
            [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0],
            [2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 0],
            [1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0],
            [2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 0],
            [2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0],
            [1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0],
            [2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 0],
            [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0],
            [1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
            [2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 3, 2, 2, 1, 1, 2, 2, 2, 0],
            [2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 3, 1, 2, 2, 2, 1, 2, 2, 2, 2, 0],
            [2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        ]

        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        score = scorer.get_score(tree, aln)
        self.assertEqual(score, 3 + 1 + 3 + 3 + 2 + 1 + 2 + 5)

    def test_get_score(self):
        aln = Align.read("TreeConstruction/msa.phy", "phylip")
        tree = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        scorer = ParsimonyScorer()
        score = scorer.get_score(tree, aln)
        self.assertEqual(score, 2 + 1 + 2 + 2 + 1 + 1 + 1 + 3)

        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2.5, 0], [2.5, 1, 0], [1, 2.5, 2.5, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        score = scorer.get_score(tree, aln)
        self.assertEqual(score, 3.5 + 2.5 + 3.5 + 3.5 + 2.5 + 1 + 2.5 + 4.5)

        alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "1",
            "2",
            "T",
            "V",
            "W",
            "Y",
            "*",
            "-",
        ]
        step_matrix = [
            [0],
            [2, 0],
            [1, 2, 0],
            [1, 2, 1, 0],
            [2, 1, 2, 2, 0],
            [1, 1, 1, 1, 2, 0],
            [2, 2, 1, 2, 2, 2, 0],
            [2, 2, 2, 2, 1, 2, 2, 0],
            [2, 2, 2, 1, 2, 2, 2, 1, 0],
            [2, 2, 2, 2, 1, 2, 1, 1, 2, 0],
            [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0],
            [2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 2, 0],
            [1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0],
            [2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 0],
            [2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0],
            [1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0],
            [2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 0],
            [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 0],
            [1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0],
            [2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 3, 2, 2, 1, 1, 2, 2, 2, 0],
            [2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 3, 1, 2, 2, 2, 1, 2, 2, 2, 2, 0],
            [2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 0],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
        ]

        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        score = scorer.get_score(tree, aln)
        self.assertEqual(score, 3 + 1 + 3 + 3 + 2 + 1 + 2 + 5)


def _fitch_cost(a, b):
    """Unit substitution cost used by the Fitch algorithm."""
    return 0 if a == b else 1


def _matrix_cost(matrix):
    """Return a substitution cost function for a matrix-like object."""

    def cost(a, b):
        return matrix[a, b]

    return cost


def _brute_force_states(tree, column, alphabet, cost):
    """Ground truth: enumerate all ancestral state assignments.

    Return the minimum parsimony score of the column and, for each
    nonterminal clade, the set of states the clade takes in at least one
    globally optimal reconstruction.
    """
    terms = tree.get_terminals()
    terms.sort(key=lambda term: term.name)
    leaf_letter = dict(zip(terms, column))
    internals = tree.get_nonterminals(order="postorder")
    best = None
    states = {clade: set() for clade in internals}
    for assignment in product(alphabet, repeat=len(internals)):
        state = dict(zip(internals, assignment))
        score = 0
        for clade in internals:
            parent = state[clade]
            for child in clade.clades:
                if child.is_terminal():
                    letter = leaf_letter[child]
                    if letter in alphabet:
                        score += cost(parent, letter)
                    else:  # unknown character: any state at no cost
                        score += min(cost(parent, a) for a in alphabet)
                else:
                    score += cost(parent, state[child])
        if best is None or score < best:
            best = score
            states = {clade: {state[clade]} for clade in internals}
        elif score == best:
            for clade in internals:
                states[clade].add(state[clade])
    return best, states


def _make_alignment(names, columns):
    """Create a MultipleSeqAlignment from names and a list of columns."""
    records = [
        SeqRecord(Seq("".join(chars)), id=name)
        for name, chars in zip(names, zip(*columns))
    ]
    return MultipleSeqAlignment(records)


def _read_rooted(newick):
    """Read a newick string and declare the tree rooted at its root."""
    tree = Phylo.read(StringIO(newick), "newick")
    tree.rooted = True
    return tree


class ParsimonyScorerStatesTest(unittest.TestCase):
    """Test ParsimonyScorer.get_states against brute force enumeration."""

    def assert_states(self, scorer, tree, alignment, alphabet=None, cost=None):
        """Check the get_states result of every column against brute force.

        Return the sum of the column scores for comparison with get_score.
        """
        results = scorer.get_states(tree, alignment)
        self.assertEqual(len(results), len(alignment[0]))
        total = 0
        for i, (score, states) in enumerate(results):
            column = alignment[:, i]
            if alphabet is None:  # Fitch algorithm
                column_alphabet = sorted(set(column))
                column_cost = _fitch_cost
            else:  # Sankoff algorithm
                column_alphabet = alphabet
                column_cost = cost
            best, expected = _brute_force_states(
                tree, column, column_alphabet, column_cost
            )
            self.assertEqual(score, best)
            self.assertEqual(set(states), set(expected))
            for clade, clade_states in states.items():
                self.assertEqual(clade_states, expected[clade])
            total += score
        return total

    def test_fitch(self):
        tree = _read_rooted("((Alpha,Beta)I1,(Gamma,(Delta,Epsilon)I2)I3)I4;")
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        columns = ["AACGG", "ACGTA", "GGGGG", "ACACA", "TGCAT", "AAGGC"]
        aln = _make_alignment(names, columns)
        scorer = ParsimonyScorer()
        total = self.assert_states(scorer, tree, aln)
        self.assertEqual(total, scorer.get_score(tree, aln))

    def test_symmetric_matrix(self):
        tree = _read_rooted("((Alpha,Beta)I1,(Gamma,(Delta,Epsilon)I2)I3)I4;")
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        columns = ["AACGG", "ACGTA", "GGGGG", "ACACA", "TGCAT", "AAGGC"]
        aln = _make_alignment(names, columns)
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2, 0], [2, 1, 0], [1, 2, 2, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        total = self.assert_states(scorer, tree, aln, alphabet, _matrix_cost(matrix))
        self.assertEqual(total, scorer.get_score(tree, aln))

    def test_asymmetric_matrix(self):
        tree = _read_rooted("((Alpha,Beta)I1,(Gamma,(Delta,Epsilon)I2)I3)I4;")
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        columns = ["AACGG", "ACGTA", "GGGGG", "ACACA", "TGCAT", "AAGGC"]
        aln = _make_alignment(names, columns)
        matrix = substitution_matrices.Array("ACGT", dims=2)
        for a in "ACGT":
            for b in "ACGT":
                matrix[a, b] = 0
        for (a, b), value in {
            ("A", "C"): 3,
            ("C", "A"): 1,
            ("A", "G"): 2,
            ("G", "A"): 4,
            ("A", "T"): 1,
            ("T", "A"): 2,
            ("C", "G"): 5,
            ("G", "C"): 1,
            ("C", "T"): 2,
            ("T", "C"): 3,
            ("G", "T"): 1,
            ("T", "G"): 2,
        }.items():
            matrix[a, b] = value
        scorer = ParsimonyScorer(matrix)
        alphabet = list(matrix.alphabet)
        total = self.assert_states(scorer, tree, aln, alphabet, _matrix_cost(matrix))
        self.assertEqual(total, scorer.get_score(tree, aln))

    def test_unknown_characters(self):
        tree = _read_rooted("((Alpha,Beta)I1,(Gamma,(Delta,Epsilon)I2)I3)I4;")
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        # 'N', '?' and '-' are not part of the matrix alphabet
        columns = ["NACGG", "AC?TA", "GGG-G", "NCN?A", "T-NT?", "NNNNN"]
        aln = _make_alignment(names, columns)
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2, 0], [2, 1, 0], [1, 2, 2, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        self.assert_states(scorer, tree, aln, alphabet, _matrix_cost(matrix))

    def test_gap_as_state(self):
        tree = _read_rooted("((Alpha,Beta)I1,(Gamma,(Delta,Epsilon)I2)I3)I4;")
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        columns = ["-ACGG", "AC-TA", "GGG-G", "A--CA", "TGCAT", "--GG-"]
        aln = _make_alignment(names, columns)
        # the caller decides to treat the gap as an ordinary state
        alphabet = ["A", "T", "C", "G", "-"]
        step_matrix = [[0], [2, 0], [2, 1, 0], [1, 2, 2, 0], [3, 3, 3, 3, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        self.assert_states(scorer, tree, aln, alphabet, _matrix_cost(matrix))

    def test_multifurcating(self):
        trees = [
            "((Alpha,Beta,Gamma)I1,(Delta,Epsilon)I2)I3;",
            "(Alpha,Beta,Gamma,Delta)I1;",
            "((Alpha,Beta)I1,(Gamma,Delta,Epsilon)I2,Eta)I3;",
        ]
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2, 0], [2, 1, 0], [1, 2, 2, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        for newick in trees:
            tree = Phylo.read(StringIO(newick), "newick")
            names = sorted(term.name for term in tree.get_terminals())
            columns = [
                "AACGGT"[: len(names)],
                "ACGTAC"[: len(names)],
                "GGGGGG"[: len(names)],
                "ACACAA"[: len(names)],
                "TGCATG"[: len(names)],
            ]
            for scorer in (ParsimonyScorer(), ParsimonyScorer(matrix)):
                if scorer.matrix is None:
                    aln = _make_alignment(names, columns)
                    self.assert_states(scorer, tree, aln)
                else:
                    aln = _make_alignment(names, columns)
                    self.assert_states(
                        scorer, tree, aln, alphabet, _matrix_cost(matrix)
                    )

    def test_duplicate_names(self):
        # the two internal clades are both named X; the terminal clades
        # share names with each other as well
        tree = _read_rooted("((A,A)X,(B,B)X)R;")
        names = ["A", "A", "B", "B"]
        columns = ["ACGT", "AGCT", "AAAA"]
        aln = _make_alignment(names, columns)
        scorer = ParsimonyScorer()
        results = scorer.get_states(tree, alignment=aln)
        for score, states in results:
            # clades are dictionary keys, so the two X clades are distinct
            self.assertEqual(len(states), 3)
            self.assertEqual(len({id(clade) for clade in states}), 3)
        total = self.assert_states(scorer, tree, aln)
        self.assertEqual(total, scorer.get_score(tree, aln))

    def test_alignment_class(self):
        tree = _read_rooted("((Alpha,Beta)I1,(Gamma,(Delta,Epsilon)I2)I3)I4;")
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
        columns = ["AACGG", "ACGTA", "GGGGG", "ACACA", "TGCAT", "AAGGC"]
        records = [
            SeqRecord(Seq("".join(chars)), id=name)
            for name, chars in zip(names, zip(*columns))
        ]
        aln = Alignment(records)
        scorer = ParsimonyScorer()
        total = self.assert_states(scorer, tree, aln)
        self.assertEqual(total, scorer.get_score(tree, aln))

    def test_random(self):
        rng = random.Random(42)
        trees = [
            "((A,B)I1,(C,(D,E)I2)I3)I4;",
            "((A,B,C)I1,(D,E)I2)I3;",
            "(A,(B,C)I1,(D,E)I2)I3;",
            "((A,B)I1,(C,D)I2,(E,F)I3)I4;",
            "(A,B,(C,D,E)I1)I2;",
        ]
        alphabet = "ACG"
        for newick in trees:
            tree = Phylo.read(StringIO(newick), "newick")
            names = sorted(term.name for term in tree.get_terminals())
            matrix = substitution_matrices.Array(alphabet, dims=2)
            for a in alphabet:
                for b in alphabet:
                    matrix[a, b] = 0 if a == b else rng.randint(1, 5)
            scorer = ParsimonyScorer(matrix)
            for _ in range(10):
                columns = [
                    "".join(rng.choice(alphabet + "N") for _ in range(len(names)))
                    for _ in range(5)
                ]
                aln = _make_alignment(names, columns)
                self.assert_states(
                    scorer, tree, aln, list(alphabet), _matrix_cost(matrix)
                )

    def test_large_tree(self):
        # the dynamic programming tables are shared between clades, so
        # larger trees are handled without enumerating reconstructions
        rng = random.Random(7)
        names = [f"Taxon{i:02d}" for i in range(40)]
        clades = [BaseTree.Clade(name=name) for name in names]
        rng.shuffle(clades)
        while len(clades) > 1:
            child1 = clades.pop(rng.randrange(len(clades)))
            child2 = clades.pop(rng.randrange(len(clades)))
            clades.append(BaseTree.Clade(clades=[child1, child2]))
        tree = BaseTree.Tree(clades[0])
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2, 0], [2, 1, 0], [1, 2, 2, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        # no unknown characters here, so that get_score can be used as reference
        columns = ["".join(rng.choice("ACGT") for _ in names) for _ in range(30)]
        aln = _make_alignment(names, columns)
        scorer = ParsimonyScorer(matrix)
        results = scorer.get_states(tree, aln)
        self.assertEqual(len(results), 30)
        for score, states in results:
            self.assertEqual(len(states), 39)
            for clade_states in states.values():
                self.assertTrue(clade_states)
                self.assertLessEqual(clade_states, set(alphabet))
        total = sum(score for score, states in results)
        self.assertEqual(total, scorer.get_score(tree, aln))

    def test_name_mismatch(self):
        tree = Phylo.read(StringIO("((A,B)I1,C)I2;"), "newick")
        aln = _make_alignment(["A", "B", "D"], ["ACG", "AGT"])
        scorer = ParsimonyScorer()
        self.assertRaises(ValueError, scorer.get_states, tree, aln)


class NNITreeSearcherTest(unittest.TestCase):
    """Test NNITreeSearcher."""

    def test_get_neighbors(self):
        tree = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2.5, 0], [2.5, 1, 0], [1, 2.5, 2.5, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        searcher = NNITreeSearcher(scorer)
        trees = searcher._get_neighbors(tree)
        self.assertEqual(len(trees), 2 * (5 - 3))
        Phylo.write(trees, os.path.join(temp_dir, "neighbor_trees.tre"), "newick")


class ParsimonyTreeConstructorTest(unittest.TestCase):
    """Test ParsimonyTreeConstructor."""

    def test_build_tree_msa(self):
        aln = AlignIO.read("TreeConstruction/msa.phy", "phylip")
        tree1 = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        tree2 = Phylo.read("./TreeConstruction/nj.tre", "newick")
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2.5, 0], [2.5, 1, 0], [1, 2.5, 2.5, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        searcher = NNITreeSearcher(scorer)
        constructor = ParsimonyTreeConstructor(searcher, tree1)
        best_tree = constructor.build_tree(aln)
        Phylo.write(best_tree, os.path.join(temp_dir, "pars1.tre"), "newick")
        constructor.starting_tree = tree2
        best_tree = constructor.build_tree(aln)
        Phylo.write(best_tree, os.path.join(temp_dir, "pars2.tre"), "newick")
        constructor.starting_tree = None
        best_tree = constructor.build_tree(aln)
        Phylo.write(best_tree, os.path.join(temp_dir, "pars3.tre"), "newick")

    def test_build_tree(self):
        aln = Align.read("TreeConstruction/msa.phy", "phylip")
        tree1 = Phylo.read("./TreeConstruction/upgma.tre", "newick")
        tree2 = Phylo.read("./TreeConstruction/nj.tre", "newick")
        alphabet = ["A", "T", "C", "G"]
        step_matrix = [[0], [2.5, 0], [2.5, 1, 0], [1, 2.5, 2.5, 0]]
        matrix = _Matrix(alphabet, step_matrix)
        scorer = ParsimonyScorer(matrix)
        searcher = NNITreeSearcher(scorer)
        constructor = ParsimonyTreeConstructor(searcher, tree1)
        best_tree = constructor.build_tree(aln)
        Phylo.write(best_tree, os.path.join(temp_dir, "pars1.tre"), "newick")
        constructor.starting_tree = tree2
        best_tree = constructor.build_tree(aln)
        Phylo.write(best_tree, os.path.join(temp_dir, "pars2.tre"), "newick")
        constructor.starting_tree = None
        best_tree = constructor.build_tree(aln)
        Phylo.write(best_tree, os.path.join(temp_dir, "pars3.tre"), "newick")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
