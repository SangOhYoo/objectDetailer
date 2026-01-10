
import unittest
import numpy as np
from core.box_sorter import sort_boxes

class TestBoxSorter(unittest.TestCase):
    def setUp(self):
        # boxes format: [x1, y1, x2, y2]
        self.boxes = np.array([
            [10, 10, 50, 50],   # Box A: Top-Left
            [60, 10, 100, 50],  # Box B: Top-Right
            [10, 60, 50, 100],  # Box C: Bottom-Left
            [30, 30, 70, 70]    # Box D: Center
        ])
        self.scores = np.array([0.9, 0.8, 0.7, 0.95])
        self.width = 110
        self.height = 110

    def test_sort_left_to_right(self):
        sorted_boxes, _, _ = sort_boxes(self.boxes, self.scores, "위치(좌에서 우)", self.width, self.height)
        # Expected order x1s: 10 (A), 10 (C), 30 (D), 60 (B)
        # Stable sort isn't guaranteed for equal keys, but A and C have same x1. 
        # We check first vs last mainly.
        self.assertEqual(sorted_boxes[0][0], 10)
        self.assertEqual(sorted_boxes[-1][0], 60)

    def test_sort_confidence(self):
        sorted_boxes, sorted_scores, _ = sort_boxes(self.boxes, self.scores, "신뢰도", self.width, self.height)
        # Expected scores: 0.95, 0.9, 0.8, 0.7
        self.assertEqual(sorted_scores[0], 0.95)
        self.assertEqual(sorted_scores[-1], 0.7)

    def test_sort_center_out(self):
        # Image center is 55, 55
        # Centers:
        # A: 30, 30 -> dist sq = (25)^2 + (25)^2 = 1250
        # B: 80, 30 -> dist sq = (25)^2 + (25)^2 = 1250
        # C: 30, 80 -> dist sq = (25)^2 + (25)^2 = 1250
        # D: 50, 50 -> dist sq = (5)^2 + (5)^2 = 50
        
        sorted_boxes, _, _ = sort_boxes(self.boxes, self.scores, "위치 (중앙에서 바깥)", self.width, self.height)
        # D should be first (closest to center)
        np.testing.assert_array_equal(sorted_boxes[0], [30, 30, 70, 70])

    def test_sort_area(self):
        # Areas:
        # A: 40*40 = 1600
        # B: 40*40 = 1600
        # C: 40*40 = 1600
        # D: 40*40 = 1600
        # All equal, add a small box
        small_box = np.array([[0,0,10,10]]) # Area 100
        all_boxes = np.vstack([self.boxes, small_box])
        all_scores = np.append(self.scores, 0.5)
        
        sorted_boxes, _, _ = sort_boxes(all_boxes, all_scores, "영역 (대형에서 소형)", self.width, self.height)
        
        # Small box should be last
        np.testing.assert_array_equal(sorted_boxes[-1], [0, 0, 10, 10])

    def test_sort_top_to_bottom(self):
        sorted_boxes, _, _ = sort_boxes(self.boxes, self.scores, "위치(위에서 아래)", self.width, self.height)
        # Expected order y1s: 10 (A), 10 (B), 30 (D), 60 (C)
        self.assertEqual(sorted_boxes[0][1], 10)
        self.assertEqual(sorted_boxes[-1][1], 60)

    def test_sort_right_to_left(self):
        sorted_boxes, _, _ = sort_boxes(self.boxes, self.scores, "위치(우에서 좌)", self.width, self.height)
        # Expected order x1s (Same as LR but reversed?): 60 (B), 30 (D), 10 (A or C)
        self.assertEqual(sorted_boxes[0][0], 60)
        self.assertEqual(sorted_boxes[-1][0], 10)

    def test_sort_bottom_to_top(self):
        sorted_boxes, _, _ = sort_boxes(self.boxes, self.scores, "위치(아래에서 위)", self.width, self.height)
        # Expected order y1s (Desc): 60 (C), 30 (D), 10 (A or B)
        self.assertEqual(sorted_boxes[0][1], 60)
        self.assertEqual(sorted_boxes[-1][1], 10)

if __name__ == '__main__':
    unittest.main()
