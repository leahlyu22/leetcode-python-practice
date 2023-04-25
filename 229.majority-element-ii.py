#
# @lc app=leetcode id=229 lang=python3
#
# [229] Majority Element II
#

# @lc code=start
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        f = len(nums) / 3
        res = set()
        # use extra memory
        hashmap = {}
        for n in nums:
            hashmap[n] = hashmap.get(n, 0) + 1
            if hashmap[n] > f:
                res.add(n)
        
        return res

        
        
# @lc code=end

