#
# @lc app=leetcode id=645 lang=python3
#
# [645] Set Mismatch
#

# @lc code=start
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        hashmap = {n: 0 for n in range(1, len(nums)+1)}
        for n in nums:
            hashmap[n] = hashmap.get(n, 0) + 1
        
        res = []
        for n, cnt in hashmap.items():
            if cnt == 2:
                res.append(n)
                break
            
        for n, cnt in hashmap.items():
            if cnt == 0:
                res.append(n)
                break
        
        return res
        
# @lc code=end

