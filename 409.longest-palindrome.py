#
# @lc app=leetcode id=409 lang=python3
#
# [409] Longest Palindrome
#

# @lc code=start
class Solution:
    def longestPalindrome(self, s: str) -> int:
        hashmap = {}
        for c in s:
            hashmap[c] = hashmap.get(c, 0) + 1
        
        res = 0
        maxOddL = 0
        for c, cnt in hashmap.items():
            res += (cnt//2 *2)
        
        for c, cnt in hashmap.items():
            if cnt % 2:
                res += 1
                break
        return res
# @lc code=end

