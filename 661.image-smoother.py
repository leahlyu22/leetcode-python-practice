#
# @lc app=leetcode id=661 lang=python3
#
# [661] Image Smoother
#

# @lc code=start
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        R, C = len(img), len(img[0]) 
        res = [[-1 for c in range(C)] for r in range(R)]

        def get_filter(i, j):
            startR = i - 1 if (i - 1) >= 0 else i
            startC = j - 1 if (j - 1) >= 0 else j
            endR = i + 1 if (i + 1) < R else R - 1
            endC = j + 1 if (j + 1) < C else C - 1
            return [startR, startC, endR, endC]
        
        def get_avg(startR, startC, endR, endC):
            smooth = 0
            for r in range(startR, endR + 1):
                for c in range(startC, endC + 1):
                    smooth += img[r][c]
            cnt = (endR - startR + 1) * (endC - startC + 1)
            return smooth // cnt
        
        for r in range(R):
            for c in range(C):
                startR, startC, endR, endC = get_filter(r, c)
                res[r][c] = get_avg(startR, startC, endR, endC)
        
        return res



            


# @lc code=end

