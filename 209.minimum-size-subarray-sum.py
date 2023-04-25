#
# @lc app=leetcode id=209 lang=python3
#
# [209] Minimum Size Subarray Sum
#

# @lc code=start
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        res = float("inf")
        l, r = 0, 0
        curSum = nums[l]

        while r < len(nums):
            if curSum < target:
                r += 1
                if r < len(nums):
                    curSum += nums[r]
                # else:
                #     break
            elif curSum > target:
                res = min(res, r - l + 1)
                curSum -= nums[l]
                l += 1
            else:
                res = min(res, r - l + 1)
                curSum -= nums[l]
                l += 1
                r += 1
                if r < len(nums):
                    curSum += nums[r]
                # else:
                #     break
                
        return res if res != float('inf') else 0

# @lc code=end

