#
# @lc app=leetcode id=16 lang=python3
#
# [16] 3Sum Closest
#

# @lc code=start
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        # initialize the result
        res = 0
        nums.sort()
        minGap = float('inf')

        for i, n in enumerate(nums):
            if i > 0 and n == nums[i - 1]:
                continue   
            l = i + 1
            r = len(nums) - 1
            while l < r:
                curSum = n + nums[l] + nums[r]
                curGap = curSum - target
                if abs(curGap) < minGap:
                    res = curSum
                    minGap = abs(curGap)
 
                
                if curGap < 0:
                    # curSum is small, increase
                    l += 1
                    while nums[l - 1] == nums[l] and l < r:
                        l += 1
                else:
                    r -= 1
        
        return res

                


        
# @lc code=end

