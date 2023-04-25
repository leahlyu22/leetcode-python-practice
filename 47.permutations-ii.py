#
# @lc app=leetcode id=47 lang=python3
#
# [47] Permutations II
#

# @lc code=start
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        perm = []
        # create a hashmap to count nums
        cnt = {}
        for n in nums:
            cnt[n] = cnt.get(n, 0) + 1
        
        def dfs():
            if len(perm) == len(nums):
                res.append(perm.copy())
                return
            
            for n in cnt:
                if cnt[n] > 0:
                    perm.append(n)
                    cnt[n] -= 1

                    dfs()
                    cnt[n] += 1
                    perm.pop()
        
        dfs()
        return res

# @lc code=end

