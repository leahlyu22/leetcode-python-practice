#
# @lc app=leetcode id=168 lang=python3
#
# [168] Excel Sheet Column Title
#

# @lc code=start
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        title = ""
        while columnNumber > 0:
            columnNumber -= 1
            title = chr(columnNumber % 26 + ord("A")) + title
            columnNumber = columnNumber // 26

        return title 
# @lc code=end

