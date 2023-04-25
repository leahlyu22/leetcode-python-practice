#
# @lc app=leetcode id=242 lang=python3
#
# [242] Valid Anagram
#

# @lc code=start
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # check the length first
        if len(s) != len(t):
            return False
        # use hashmap
        mapS, mapT = {}, {}
        for c in s:
            mapS[c] = mapS.get(c, 0) + 1
        for c in t:
            mapT[c] = mapT.get(c, 0) + 1

        for key, val in mapS.items():
            if key not in mapT or val != mapT[key]:
                return False
        return True
            

            
        
# @lc code=end

