#
# @lc app=leetcode id=71 lang=python3
#
# [71] Simplify Path
#

# @lc code=start
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []

        for i in path.split("/"):
            if i == "..":
                if stack:
                    stack.pop()
            elif i == "." or i == "":
                continue
            else:
                stack.append(i)
        
        return '/' + '/'.join(stack)

        
# @lc code=end

