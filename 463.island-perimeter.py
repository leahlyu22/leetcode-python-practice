#
# @lc app=leetcode id=463 lang=python3
#
# [463] Island Perimeter
#

# @lc code=start
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        perimeter = 0
        rows, cols = len(grid), len(grid[0])
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # check its neighbors
                    if r == 0 or grid[r-1][c] == 0:
                        perimeter += 1
                    if c == 0 or grid[r][c - 1] == 0:
                        perimeter += 1
                    if r == rows - 1 or grid[r+1][c] == 0:
                        perimeter += 1
                    if c == cols - 1 or grid[r][c + 1] == 0:
                        perimeter += 1
        
        return perimeter

# @lc code=end

