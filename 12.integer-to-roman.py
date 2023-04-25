#
# @lc app=leetcode id=12 lang=python3
#
# [12] Integer to Roman
#

# @lc code=start
class Solution:
    def intToRoman(self, num: int) -> str:
        symList = [['I', 1],
                   ['IV', 4],
                   ['V', 5],
                   ['IX', 9],
                   ['X', 10],
                   ['XL', 40],
                   ['L', 50],
                   ['XC', 90],
                   ['C', 100],
                   ['CD', 400],
                   ['D', 500],
                   ['CM', 900],
                   ['M', 1000]]
        
        res = ""
        for sym, val in reversed(symList):
            if num // val:
                cnt = num // val
                res += cnt * sym
                num = num % val
        
        return res

# @lc code=end

