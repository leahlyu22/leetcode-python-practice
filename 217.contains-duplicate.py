#
# @lc app=leetcode id=217 lang=python3
#
# [217] Contains Duplicate
#

# @lc code=start
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashmap = {}
        for n in nums:
            if n in hashmap:
                return True
            hashmap[n] = hashmap.get(n, 0) + 1
        
        return False
        
# @lc code=end

