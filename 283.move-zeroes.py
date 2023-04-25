#
# @lc app=leetcode id=283 lang=python3
#
# [283] Move Zeroes
#

# @lc code=start
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        i = j = 0
        while j < len(nums):
            if nums[i] != 0 and nums[j] != 0:
                i += 1
                j += 1
            else:
                while j < len(nums) and nums[j] == 0 :
                    j += 1
                if j < len(nums):
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j += 1
        

        
        
# @lc code=end

