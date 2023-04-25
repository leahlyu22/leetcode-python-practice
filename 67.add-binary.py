#
# @lc app=leetcode id=67 lang=python3
#
# [67] Add Binary
#

# @lc code=start
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        res = ""
        carry = 0

        a = a[::-1]
        b = b[::-1]
        for i in range(max(len(a), len(b))):
            bitA = ord(a[i]) - ord('0') if i < len(a) else 0
            bitB = ord(b[i]) - ord('0') if i < len(b) else 0

            total = bitA + bitB + carry
            char = str(total % 2)
            res = char + res
            carry = total // 2
        
        if carry:
            res = "1" + res

        return res
            
        
# @lc code=end

