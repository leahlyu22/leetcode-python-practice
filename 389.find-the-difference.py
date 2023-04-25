#
# @lc app=leetcode id=389 lang=python3
#
# [389] Find the Difference
#

# @lc code=start
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        hashmap = {}
        for c in s:
            hashmap[c] = hashmap.get(c, 0) + 1
        
        for c in t:
            if c not in hashmap:
                return c
            else:
                hashmap[c] -= 1
        
        for c, cnt in hashmap.items():
            if cnt != 0:
                return c
        
        
# @lc code=end

