#
# @lc app=leetcode id=374 lang=python3
#
# [374] Guess Number Higher or Lower
#

# @lc code=start
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if num is higher than the picked number
#          1 if num is lower than the picked number
#          otherwise return 0
# def guess(num: int) -> int:

class Solution:
    def guessNumber(self, n: int) -> int:
        l = 1
        r = n
        while l <= r:
            pivot = l + (r - l) // 2
            if guess(pivot) == -1:
                # too large
                r = pivot - 1
            elif guess(pivot) == 1:
                l = pivot + 1
            else:
                return pivot
# @lc code=end

