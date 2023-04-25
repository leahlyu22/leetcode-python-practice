#
# @lc app=leetcode id=231 lang=python3
#
# [231] Power of Two
#

# @lc code=start
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        # base case
        if n == 1:
            return True
        if n == 0:
            return False
        if n % 2:
            return False

        return self.isPowerOfTwo(n/2)
        
# @lc code=end

