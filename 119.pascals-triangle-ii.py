#
# @lc app=leetcode id=119 lang=python3
#
# [119] Pascal's Triangle II
#

# @lc code=start
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        prevR = [1]

        for r in range(rowIndex + 1):
            curR = [1] * (r + 1)
            if r > 1:
                for i in range(1, r):
                    curR[i] = prevR[i-1] + prevR[i]
            prevR = curR
        
        return curR
        
# @lc code=end

