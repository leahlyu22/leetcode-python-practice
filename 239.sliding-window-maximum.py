#
# @lc app=leetcode id=239 lang=python3
#
# [239] Sliding Window Maximum
#

# @lc code=start
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        q = collections.deque()
        l, r = 0, 0

        while r < len(nums):
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            # remove out-of-bound value
            if l > q[0]:
                q.popleft()

            # append maximum value in a window
            if r + 1 >= k:
                output.append(nums[q[0]])
                l += 1
            r += 1
        
        return output

# @lc code=end

