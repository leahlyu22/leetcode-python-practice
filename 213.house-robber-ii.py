#
# @lc app=leetcode id=213 lang=python3
#
# [213] House Robber II
#

# @lc code=start
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        def helper(nums):
            dp = [0, 0] + nums
            for i in range(2, len(dp)):
                dp[i] = max(dp[i-2]+dp[i], dp[i-1])
            
            return dp[-1]
    
        return max(helper(nums[1:]), helper(nums[:-1]))
        
        


        
# @lc code=end

