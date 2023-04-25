#
# @lc app=leetcode id=326 lang=python3
#
# [326] Power of Three
#

# @lc code=start
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n == 0:
            return False
        # base condition
        if n == 1:
            return True
        if n % 3:
            return False
        return self.isPowerOfThree(n/3)


        
# @lc code=end

