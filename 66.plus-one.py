#
# @lc app=leetcode id=66 lang=python3
#
# [66] Plus One
#

# @lc code=start
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        # initialize carry
        carry = 1
        res = []
        for i in range(len(digits)-1, -1, -1):
            digit = digits[i] + carry
            res.append(digit % 10)
            carry = digit // 10
        
        if carry == 1:
            res.append(1)
        return res[::-1]
        
# @lc code=end
