#
# @lc app=leetcode id=378 lang=python3
#
# [378] Kth Smallest Element in a Sorted Matrix
#

# @lc code=start
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        rows, cols = len(matrix), len(matrix[0])
        t, b = 0, rows - 1
        l, r = 0, cols - 1
        

        # find the correct row first
        while t <= b:
            midR = (t + b) // 2
            minL = t * cols + 1
            maxR = (t + 1) * cols
            if k < minL:
                b = midR - 1
            elif k > maxR:
                t = midR + 1
            else:
                break
        
        if t <= b and l <= r:
            row = midR
        
        # find the correct col
        while l <= r:
            midC = (l + r) // 2
            kth = minL + 1
            if kth < k:
                l = midC + 1
            elif kth > k:
                r = midC - 1
            else:
                break
        if t <= b and l <= r:
            col = midC
            return matrix[row][col]
        else:
            return matrix[0][0]


# @lc code=end

