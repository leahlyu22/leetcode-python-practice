#
# @lc app=leetcode id=69 lang=python3
#
# [69] Sqrt(x)
#

# @lc code=start
class Solution:
    def mySqrt(self, x: int) -> int:
        l = 0
        r = x
        while l <= r:
            mid = l + (r-l)//2
            if mid * mid <= x < (mid + 1) * (mid + 1):
                return mid
            elif mid * mid > x:
                r = mid - 1
            else:
                l = mid + 1
        # l, r = 0, x
        # while l <= r:
        #     mid = l + (r-l)//2
        #     if mid * mid <= x < (mid + 1) * (mid + 1):
        #         return mid
        #     elif x > mid * mid:
        #         l = mid + 1
        #     else:
        #         r = mid - 1

        
        
# @lc code=end

