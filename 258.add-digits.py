#
# @lc app=leetcode id=258 lang=python3
#
# [258] Add Digits
#

# @lc code=start
class Solution:
    def addDigits(self, num: int) -> int:
        
        def helper(num):
            # return res -> the new number
            res = 0
            while num:
                digit = num % 10
                res += digit
                num = num // 10
            return res
        
        while num // 10:
            num = helper(num)
        return num

# @lc code=end

