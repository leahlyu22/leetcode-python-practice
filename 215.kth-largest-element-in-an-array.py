#
# @lc app=leetcode id=215 lang=python3
#
# [215] Kth Largest Element in an Array
#

# @lc code=start
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        k = len(nums) - k
        l, r = 0, len(nums) - 1
        while l < r:
            p = self.quickSelect(nums, l, r)
            if p > k:
                r = p - 1
            elif p < k:
                l = p + 1
            else:
                break
        
        return nums[k]

    def quickSelect(self, nums, left, right):
        # return the pivot idx
        pivot = nums[right]
        p = left # pointer for pivot

        for i in range(left, right):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[right] = nums[right], nums[p]
        return p


        
        
# @lc code=end

