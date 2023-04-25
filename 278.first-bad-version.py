#
# @lc app=leetcode id=278 lang=python3
#
# [278] First Bad Version
#

# @lc code=start
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:

class Solution:
    def firstBadVersion(self, n: int) -> int:
        l = 1
        r = n
        while l <= r:
            mid = (r + l) // 2
            if isBadVersion(mid) is True:
                r = mid - 1
            else:
                l = mid + 1
            
            if isBadVersion(l) is True:
                return l
        
# @lc code=end

