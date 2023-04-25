#
# @lc app=leetcode id=81 lang=python3
#
# [81] Search in Rotated Sorted Array II
#

# @lc code=start
# class Solution:
#     def search(self, nums: List[int], target: int) -> bool:
#         l, r = 0, len(nums) - 1
#         while l <= r:
#             m = (l + r) // 2
#             if nums[m] == target:
#                 return True
            
#             if nums[m] >= nums[l]:
#                 if target > nums[m] or target < nums[l]:
#                     l = m + 1
#                 else:
#                     r = m - 1
#             else:
#                 if target < nums[m] or target > nums[r]:
#                     r = m - 1
#                 else:
#                     l = m + 1
#         return False
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (l + r) // 2
            if target == nums[mid]:
                return True

            # left sorted portion
            if nums[l] <= nums[mid]:
                if target > nums[mid] or target < nums[l]:
                    l = mid + 1
                else:
                    r = mid - 1
            # right sorted portion
            else:
                if target < nums[mid] or target > nums[r]:
                    r = mid - 1
                else:
                    l = mid + 1
        return False







        
# @lc code=end

