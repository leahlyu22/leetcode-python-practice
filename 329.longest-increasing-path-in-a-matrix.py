#
# @lc app=leetcode id=329 lang=python3
#
# [329] Longest Increasing Path in a Matrix
#

# @lc code=start
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        dp = {}
        rows, cols = len(matrix), len(matrix[0])

        def dfs(r, c, preV):
            if (r not in range(rows) or
                c not in range(cols) or
                matrix[r][c] <= preV):
                return 0 
            if (r, c) in dp:
                return dp[(r, c)]
                
            res = 1
            res = max(res, 1 + dfs(r + 1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r - 1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))
            res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))

            dp[(r, c)] = res
            return res
    
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, -1)
        
        return max(dp.values())

# @lc code=end

