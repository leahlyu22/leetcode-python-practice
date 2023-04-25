#
# @lc app=leetcode id=202 lang=python
#
# [202] Happy Number
#

# @lc code=start
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        def sumSquare(n):
            res = 0
            while n:
                digit = n % 10
                res += (digit ** 2)
                n = n // 10
            return res

        res = set()
        while True:
            n = sumSquare(n)
            if n == 1:
                return True
            if n in res:
                return False
            res.add(n)
        
# @lc code=end

