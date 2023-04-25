#
# @lc app=leetcode id=345 lang=python3
#
# [345] Reverse Vowels of a String
#

# @lc code=start
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u']
        l, r = 0, len(s) - 1
        s = list(s)

        while l <= r:
            if s[l].lower() in vowels and s[r].lower() in vowels:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
            elif s[l].lower() in vowels:
                r -= 1
            elif s[r].lower() in vowels:
                l += 1
            else:
                l += 1
                r -= 1

        return "".join(s)
            

# @lc code=end

