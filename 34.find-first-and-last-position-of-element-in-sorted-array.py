#
# @lc app=leetcode id=34 lang=python3
#
# [34] Find First and Last Position of Element in Sorted Array
#

# @lc code=start
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = self.binarySearch(nums, target, True)
        right = self.binarySearch(nums, target, False)
        return [left, right]


    def binarySearch(self, nums, target, left):
        boundary = -1
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (r + l)//2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                boundary = m
                if left:
                    r = m - 1
                else:
                    l = m + 1
        return boundary
# @lc code=end

