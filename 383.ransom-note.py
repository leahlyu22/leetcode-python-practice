#
# @lc app=leetcode id=383 lang=python3
#
# [383] Ransom Note
#

# @lc code=start
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        mapMag = {}
        mapNote = {}

        # create two maps
        for c in ransomNote:
            mapNote[c] = mapNote.get(c, 0) + 1
        for c in magazine:
            mapMag[c] = mapMag.get(c, 0) + 1
        
        # search
        for c in ransomNote:
            if c not in magazine or mapNote[c] > mapMag[c]:
                return False
        
        return True
        
# @lc code=end

