#
# @lc app=leetcode id=26 lang=python3
#
# [26] Remove Duplicates from Sorted Array
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        prev, cur = 0, 0
        while cur < len(nums):
            while cur < len(nums) and nums[prev] == nums[cur]:
                cur += 1
            # find the index that two numbers are not equal
            if cur >= len(nums):
                break
            if prev + 1 != cur:
                temp = nums[prev + 1]
                nums[prev + 1] = nums[cur]
                nums[cur] = temp
            prev += 1
            cur += 1
        
        return prev + 1

        
# @lc code=end

