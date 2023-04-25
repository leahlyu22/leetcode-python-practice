#
# @lc app=leetcode id=414 lang=python3
#
# [414] Third Maximum Number
#

# @lc code=start
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        nums = [-1 * n for n in nums]
        heapq.heapify(nums)
        k = 2
        num0 = -1 * heapq.heappop(nums)
        maxN = num0
        while nums and k:
            num1 = -1 * heapq.heappop(nums)
            if num0 == num1:
                continue
            else:
                k -= 1
                num0 = num1
        
        return num0 if k == 0 else maxN

# @lc code=end

