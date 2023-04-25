#
# @lc app=leetcode id=2013 lang=python3
#
# [2013] Detect Squares
#

# @lc code=start
class DetectSquares:

    def __init__(self):
        # create a list to store all points
        self.pts = []
        # create a hashmap to cnt number of each point
        self.ptsCnt = defaultdict(int)


    def add(self, point: List[int]) -> None:
        # add the point
        self.pts.append(point)
        self.ptsCnt[tuple(point)] += 1

    def count(self, point: List[int]) -> int:
        res = 0
        px, py = point
        # find the diagonal points from the pts list
        for x, y in self.pts:
            # diagonal points would have height == width
            if abs(px - x) != abs(py - y) or px == x or py == y:
                continue
            res += self.ptsCnt[x, py] * self.ptsCnt[px, y]
        
        return res


# Your DetectSquares object will be instantiated and called as such:
# obj = DetectSquares()
# obj.add(point)
# param_2 = obj.count(point)
# @lc code=end

