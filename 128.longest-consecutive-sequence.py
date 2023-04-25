#
# @lc app=leetcode id=128 lang=python3
#
# [128] Longest Consecutive Sequence
#

# @lc code=start
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        maxL = 0
        numSet = set(nums)
        for n in nums:
            if (n - 1) not in numSet:
                curL = 0
                while (n + curL) in numSet:
                    curL += 1
                maxL = max(maxL, curL)
        return maxL

# @lc code=end

