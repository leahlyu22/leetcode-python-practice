#
# @lc app=leetcode id=268 lang=python3
#
# [268] Missing Number
#

# @lc code=start
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = 0
        n = len(nums)
        for i in range(len(nums)):
            res += (i-nums[i])
        
        return res + n
        
# @lc code=end

