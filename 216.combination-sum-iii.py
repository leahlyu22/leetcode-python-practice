#
# @lc app=leetcode id=216 lang=python3
#
# [216] Combination Sum III
#

# @lc code=start
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        candidates = [i for i in range(1, 10)]

        def dfs(i, cur, curSum):
            if i > len(candidates) or curSum > n or len(cur) > k:
                return
            if curSum == n and len(cur) == k:
                res.append(cur.copy())
                return
            
            cur.append(i)
            dfs(i+1, cur, curSum+i)
            cur.pop()
            dfs(i+1, cur, curSum)
        
        dfs(1, [], 0)
        return res



# @lc code=end

