#
# @lc app=leetcode id=415 lang=python3
#
# [415] Add Strings
#

# @lc code=start
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        num1 = num1[::-1]
        num2 = num2[::-1]
        i, j = 0, 0
        carry = 0
        res = ""
        while i < len(num1) or j < len(num2):
            digit1 = int(num1[i]) if i < len(num1) else 0
            digit2 = int(num2[j]) if j < len(num2) else 0
            digit = digit1 + digit2 + carry
            res += str(digit % 10)
            carry = digit // 10
            i += 1
            j += 1
        
        if carry:
            res += '1'
        
        return res[::-1]
       
        
        

                
        
# @lc code=end

