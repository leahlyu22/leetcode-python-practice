#
# @lc app=leetcode id=64 lang=python3
#
# [64] Minimum Path Sum
#

# @lc code=start
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        ROW = [grid[-1][c] for c in range(cols)]
        # ROW[-1] = grid[-1][-1]

        for c in range(cols-2, -1, -1):
            ROW[c] += ROW[c+1]
        
        for r in range(rows-2, -1, -1):
            newR = [grid[r][i] for i in range(cols)]
            newR[-1] += ROW[-1]
            for c in range(cols-2, -1, -1):
                newR[c] += min(ROW[c], newR[c+1])
            ROW = newR
        
        return ROW[0]
# @lc code=end

