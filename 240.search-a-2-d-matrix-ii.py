#
# @lc app=leetcode id=240 lang=python3
#
# [240] Search a 2D Matrix II
#

# @lc code=start
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            l, r = 0, len(row) - 1
            while l <= r:
                m = (l+r)//2
                if row[m] == target:
                    return True
                elif row[m] > target:
                    r = m - 1
                else:
                    l = m + 1
        return False

# @lc code=end

