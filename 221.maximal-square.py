#
# @lc app=leetcode id=221 lang=python3
#
# [221] Maximal Square
#

# @lc code=start
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rows, cols = len(matrix), len(matrix[0])
        cache = {} # store the maxLen of square

        def dfs(r, c):
            if r not in range(rows) or c not in range(cols):
                return 0
            
            if (r, c) not in cache:
                right = dfs(r, c + 1)
                down = dfs(r + 1, c)
                diag = dfs(r + 1, c + 1)

                cache[(r, c)] = 0
                if matrix[r][c] == '1':
                    cache[(r, c)] = 1 + min(right, down, diag)
            
            return cache[(r, c)]

        dfs(0, 0)
        return max(cache.values()) ** 2

# @lc code=end

