#
# @lc app=leetcode id=205 lang=python3
#
# [205] Isomorphic Strings
#

# @lc code=start
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        mapST, mapTS = {}, {}
        for c1, c2 in zip(s, t):
            if (c1 in mapST and mapST[c1]!=c2) or (c2 in mapTS and mapTS[c2]!=c1):
                return False
            mapST[c1] = c2
            mapTS[c2] = c1
        
        return True
# @lc code=end

