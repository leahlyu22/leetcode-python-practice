#
# @lc app=leetcode id=59 lang=python3
#
# [59] Spiral Matrix II
#

# @lc code=start
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[-1 for i in range(n)] for j in range(n)]
        cnt = 1

        l, r = 0, n - 1
        t, b = 0, n - 1

        while l <= r and t <= b:
            for col in range(l, r + 1):
                matrix[t][col] = cnt
                cnt += 1
            t += 1

            for row in range(t, b+1):
                matrix[row][r] = cnt
                cnt += 1
            r -= 1
            
            if not l <= r and not t <= b:
                break

            for col in range(r, l-1, -1):
                matrix[b][col] = cnt
                cnt += 1
            b -= 1
            
            for row in range(b, t-1, -1):
                matrix[row][l] = cnt
                cnt += 1
            l += 1
        return matrix
# @lc code=end

