#
# @lc app=leetcode id=387 lang=python3
#
# [387] First Unique Character in a String
#

# @lc code=start
class Solution:
    def firstUniqChar(self, s: str) -> int:
        hashmap = {}
        for i in range(len(s)):
            if s[i] not in hashmap:
                hashmap[s[i]] = i
            else:
                hashmap[s[i]] = float('inf')
        
        idx = min(hashmap.values())
        return idx if idx != float('inf') else -1
        
# @lc code=end

