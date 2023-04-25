#
# @lc app=leetcode id=7 lang=python3
#
# [7] Reverse Integer
#

# @lc code=start
class Solution:
    def reverse(self, x: int) -> int:
        # define the boundary
        MIN = -2 ** 31
        MAX = 2 ** 31 - 1

        res = 0
        while x:
            digit = int(math.fmod(x, 10)) # get the remainder
            x = int(x / 10)

            if (res > MAX // 10 or
                (res == MAX // 10 and digit >= MAX % 10)):
                return 0
            if (res < MIN // 10 or
                (res == MIN // 10 and digit <= MAX % 10)):
                return 0
            res = res * 10 + digit
        
        return res
        
# @lc code=end

