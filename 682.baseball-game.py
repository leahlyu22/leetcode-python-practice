#
# @lc app=leetcode id=682 lang=python3
#
# [682] Baseball Game
#

# @lc code=start
class Solution:
    def calPoints(self, operations: List[str]) -> int:
        stack = []
        for record in operations:
            if record == "C" and stack:
                stack.pop()
            elif record == "D" and stack:
                stack.append(stack[-1]*2)
            elif record == "+" and stack:
                stack.append(stack[-1]+stack[-2])
            else:
                stack.append(int(record))
        
        res = 0
        while stack:
            res += stack.pop()
        return res 

# @lc code=end

