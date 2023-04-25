#
# @lc app=leetcode id=171 lang=python3
#
# [171] Excel Sheet Column Number
#

# @lc code=start
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        cycle = len(columnTitle)
        num = 0

        for s in columnTitle:
            num += (ord(s) - ord('A') + 1) * (26**(cycle-1))
            cycle -= 1
        
        return num

 # @lc code=end

