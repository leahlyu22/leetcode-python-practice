#
# @lc app=leetcode id=304 lang=python3
#
# [304] Range Sum Query 2D - Immutable
#

# @lc code=start
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        rows, cols = len(matrix), len(matrix[0])
        self.sumM = [[0] * (cols+1) for r in range(rows+1)]

        for r in range(rows):
            prefix = 0
            for c in range(cols):
                prefix += matrix[r][c]
                above = self.sumM[r][c + 1]
                self.sumM[r + 1][c + 1] = prefix + above

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        # offset
        r1, c1, r2, c2 = row1 + 1, col1 + 1, row2 + 1, col2 + 1
        
        bottomRight = self.sumM[r2][c2]
        above = self.sumM[r1-1][c2]
        left = self.sumM[r2][c1-1]
        topLeft = self.sumM[r1-1][c1-1]
        return bottomRight - above - left + topLeft

        
# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
# @lc code=end

