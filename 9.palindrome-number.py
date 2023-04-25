#
# @lc app=leetcode id=9 lang=python3
#
# [9] Palindrome Number
#

# @lc code=start
class Solution:
    def isPalindrome(self, x: int) -> bool:
        s = str(x)
        l, r = 0, len(s) - 1
        while l <= r:
            if s[l] != s[r]:
                return False
            else:
                l += 1
                r -= 1

        return True 
# @lc code=end

