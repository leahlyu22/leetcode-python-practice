#
# @lc app=leetcode id=10 lang=python3
#
# [10] Regular Expression Matching
#

# @lc code=start
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        cache = {} # (i, j) -> True/False

        def dfs(i, j):
            if i >= len(s) and j >= len(p):
                return True
            if j >= len(p):
                return False
            if (i, j) in cache:
                return cache[(i, j)]

            match = (i < len(s) and (s[i] == p[j] or p[j] == '.'))
            if (j + 1) < len(p) and p[j + 1] == '*':
                cache[(i, j)] = (dfs(i, j + 2) or 
                (match and dfs(i + 1, j)))
                
                return cache[(i, j)]
            
            if match:
                cache[(i, j)] = dfs(i + 1, j + 1)
                return cache[(i, j)]
            
            cache[(i, j)] = False
            return cache[(i, j)]
        
        return dfs(0, 0)

# @lc code=end

