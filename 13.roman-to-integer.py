#
# @lc app=leetcode id=13 lang=python3
#
# [13] Roman to Integer


# @lc code=start
class Solution:
    def romanToInt(self, s: str) -> int:
        res = 0
        romanToint = {'I': 1, 
                      'V': 5,
                      'X': 10,
                      'L': 50,
                      'C': 100,
                      'D': 500,
                      'M': 1000
                      }
        
        for i in range(len(s)):
            if i + 1 < len(s) and romanToint[s[i]] < romanToint[s[i + 1]]:
                res -= romanToint[s[i]]
            else:
                res += romanToint[s[i]]
        
        return res
# @lc code=end

