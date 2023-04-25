#
# @lc app=leetcode id=151 lang=python3
#
# [151] Reverse Words in a String
#

# @lc code=start
class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split()
        return " ".join(words[::-1])

            
            
            
# @lc code=end

