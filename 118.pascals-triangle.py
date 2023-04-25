#
# @lc app=leetcode id=118 lang=python3
#
# [118] Pascal's Triangle
#

# @lc code=start
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        prevR = [1]
        
        for r in range(1, numRows):
            curR = [1] * (r + 1)
            if r > 1:
                for i in range(1, r):
                    curR[i] = prevR[i-1] + prevR[i]
            res.append(curR)
            prevR = curR
        
        return res


                
# @lc code=end

