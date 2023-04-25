#
# @lc app=leetcode id=567 lang=python3
#
# [567] Permutation in String
#

# @lc code=start
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        
        s1Cnt = [0] * 26
        s2Cnt = [0] * 26
        for i in range(len(s1)):
            s1Cnt[ord(s1[i]) - ord('a')] += 1
            s2Cnt[ord(s2[i]) - ord('a')] += 1
        
        match = 0
        # cnt the initial match
        for i in range(26):
            match += (1 if s1Cnt[i] == s2Cnt[i] else 0)

        l = 0
        for r in range(len(s1), len(s2)):
            if match == 26:
                return True
            
            # modify the current list
            idx_r = ord(s2[r]) - ord('a')
            s2Cnt[idx_r] += 1
            if s1Cnt[idx_r] == s2Cnt[idx_r]:
                match += 1
            elif s1Cnt[idx_r] == s2Cnt[idx_r] - 1:
                match -= 1

            idx_l = ord(s2[l]) - ord('a')
            s2Cnt[idx_l] -= 1
            if s1Cnt[idx_l] == s2Cnt[idx_l]:
                match += 1
            elif s1Cnt[idx_l] == s2Cnt[idx_l] + 1:
                match -= 1
            l += 1
        
        return match == 26

# @lc code=end

