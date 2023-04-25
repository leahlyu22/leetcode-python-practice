#
# @lc app=leetcode id=347 lang=python3
#
# [347] Top K Frequent Elements
#

# @lc code=start
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        hashmap = {}
        for n in nums:
            hashmap[n] = hashmap.get(n, 0) + 1
        
        minHeap = []
        for n, cnt in hashmap.items():
            heapq.heappush(minHeap, [-cnt, n])
        
        res = []
        while k:
            res.append(heapq.heappop(minHeap)[1])
            k -= 1
        return res
# @lc code=end

