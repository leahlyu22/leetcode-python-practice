#
# @lc app=leetcode id=165 lang=python3
#
# [165] Compare Version Numbers
#

# @lc code=start
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        def findStart(s):
            i = 0
            while i < len(s):
                if s[i] == '0':
                    i += 1
                else:
                    return i
            return i


        def helper(s1, s2):
            i = findStart(s1)
            j = findStart(s2)
            if len(s1[i:]) > len(s2[j:]):
                return 1
            elif len(s1[i:]) < len(s2[j:]):
                return -1
            else:
                while i < len(s1) and j < len(s2):
                    if int(s1[i]) == int(s2[j]):
                        i += 1
                        j += 1
                    elif int(s1[i]) > int(s2[j]):
                        return 1
                    else:
                        return -1
            
                return 0 


        v1 = version1.split('.')
        v2 = version2.split('.')

        maxL = max(len(v1), len(v2))
        i = 0
        while i < maxL:
            s1 = v1[i] if i < len(v1) else '0'
            s2 = v2[i] if i < len(v2) else '0'
            res = helper(s1, s2)
            if res == 1:
                return 1
            elif res == -1:
                return -1
            else:
                i += 1
        
        return 0
# @lc code=end

