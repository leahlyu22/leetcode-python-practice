#
# @lc app=leetcode id=136 lang=python3
#
# [136] Single Number
#

# @lc code=start
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # use XOR 
        # XOR will return 1 for different, 0 for same
        res = nums[0]
        for n in nums[1:]:
            res = (res ^ n)
        
        return res
# @lc code=end

