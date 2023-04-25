#
# @lc app=leetcode id=32 lang=python3
#
# [32] Longest Valid Parentheses
#

# @lc code=start
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        dp = 0
        closeN = openN = 0

        for c in s:
            if c == "(":
                openN += 1
            else:
                closeN += 1

            if openN >= closeN:
                dp = closeN
            else:
                closeN -= 1
        
        return dp*2
        
# @lc code=end

