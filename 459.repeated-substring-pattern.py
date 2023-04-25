#
# @lc app=leetcode id=459 lang=python3
#
# [459] Repeated Substring Pattern
#

# @lc code=start
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        sLen = len(s)
        for i in range(sLen):
            if (sLen % (i + 1)) == 0:
                if i == sLen - 1:
                    return False
                else:
                    for j in range(i + 1, sLen, i + 1):
                        if s[j: j + i + 1] != s[: i + 1]:
                            break
                    if j == sLen - (i + 1) and s[j: j + i + 1] == s[: i + 1]:
                        return True
            
        return False
            
        
# @lc code=end

