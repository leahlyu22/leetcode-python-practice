#
# @lc app=leetcode id=63 lang=python3
#
# [63] Unique Paths II
#

# @lc code=start
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[-1][-1] == 1:
            return 0
        
        rows, cols = len(obstacleGrid), len(obstacleGrid[0])
        ROW = [1] * cols
        for c in range(cols-2, -1, -1):
            if obstacleGrid[-1][c] == 1 or ROW[c + 1] == -1:
                ROW[c] = -1

        for r in range(rows-2, -1, -1):
            newR = [0] * cols
            newR[-1] = 1 if (obstacleGrid[r][-1]!=1 and ROW[-1]!=-1) else -1

            for c in range(cols-2, -1, -1):
                if obstacleGrid[r][c] == 1 or (newR[c + 1]==-1 and ROW[c]==-1):
                    newR[c] = -1
                else:
                    newR[c] = newR[c+1] + ROW[c] if (newR[c+1] * ROW[c] >=0) else max(newR[c+1], ROW[c])
            ROW = newR
        
        return ROW[0] if ROW[0]!= -1 else 0
            
# @lc code=end

