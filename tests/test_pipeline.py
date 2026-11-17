"""End-to-end checks on public sample game 3.

    python -m unittest discover tests

Set SAMPLE_GAME_3 if the sample data is not in ../sample-data.
"""

import hashlib
import os
import tempfile
import unittest

import numpy as np

from possession import decode, io, pipeline

FOLDER = os.environ.get("SAMPLE_GAME_3") or os.path.join(
    os.path.dirname(io.REPO), "sample-data", "data", "Sample_Game_3")


class ViterbiTest(unittest.TestCase):
    def test_single_frame_of_evidence_does_not_switch(self):
        scores = np.zeros((500, 3))
        scores[:, decode.H] = 0.1
        scores[250] = [-2.0, 2.0, 0.0]
        self.assertTrue(np.all(decode.viterbi(scores) == decode.H))

    def test_sustained_evidence_switches(self):
        scores = np.zeros((500, 3))
        scores[:250, decode.H] = 0.3
        scores[250:, decode.A] = 0.3
        path = decode.viterbi(scores)
        self.assertEqual(path[0], decode.H)
        self.assertEqual(path[-1], decode.A)
        self.assertEqual(np.count_nonzero(np.diff(path)), 1)


class SampleGameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.match = io.load_sample(FOLDER)
        cls.codes = pipeline.label_codes(cls.match)

    def test_every_frame_in_a_half_is_labelled(self):
        for start, stop in self.match.halves:
            self.assertTrue(np.all(self.codes[start:stop] != pipeline.X))

    def test_possessions_look_like_football(self):
        change = np.flatnonzero(np.diff(self.codes)) + 1
        starts = np.concatenate(([0], change))
        spells = np.sum(self.codes[starts] < 2)
        self.assertTrue(150 < spells < 600)

    def test_output_is_reproducible(self):
        digests = []
        for _ in range(2):
            codes = pipeline.label_codes(self.match)
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "out.csv")
                pipeline.write_submission(path, codes)
                with open(path, "rb") as handle:
                    digests.append(hashlib.sha256(handle.read()).hexdigest())
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
