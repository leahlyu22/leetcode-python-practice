#
# @lc app=leetcode id=290 lang=python3
#
# [290] Word Pattern
#

# @lc code=start
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        mapP = {}
        mapS = {}
        
        word = s.split(" ")
        if len(pattern) != len(word):
            return False

        for i in range(len(pattern)):
            if pattern[i] not in mapP:
                mapP[pattern[i]] = word[i]
            if mapP[pattern[i]] != word[i]:
                return False
            
            if word[i] not in mapS:
                mapS[word[i]] = pattern[i]
            if mapS[word[i]] != pattern[i]:
                return False
        
        return True
        
# @lc code=end

