# tests/test_collision.py
import unittest
from utils import line_intersects_circle

class TestCollision(unittest.TestCase):
    def test_center_hit(self):
        p1 = (0,0); p2 = (10,0)
        center = (5,0); radius = 1
        self.assertTrue(line_intersects_circle(p1,p2,center,radius))

    def test_miss(self):
        p1 = (0,0); p2 = (10,0)
        center = (5,5); radius = 1
        self.assertFalse(line_intersects_circle(p1,p2,center,radius))

    def test_endpoint_touch(self):
        p1 = (0,0); p2 = (10,0)
        center = (10,0); radius = 0.5
        self.assertTrue(line_intersects_circle(p1,p2,center,radius))

if __name__ == "__main__":
    unittest.main()
