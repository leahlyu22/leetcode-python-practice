#
# @lc app=leetcode id=355 lang=python3
#
# [355] Design Twitter
#

# @lc code=start
class Twitter:

    def __init__(self):
        self.count = 0
        self.followMap = defaultdict(set)
        self.tweetMap = defaultdict(list)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweetMap[userId].append([self.count, tweetId])
        self.count -= 1

    def getNewsFeed(self, userId: int) -> List[int]:
        # add itself to the followMap
        self.followMap[userId].add(userId)
        res = []
        minHeap = []
        heapq.heapify(minHeap)

        for followeeId in self.followMap[userId]:
            if followeeId in self.tweetMap:
                idx = len(self.tweetMap[followeeId]) - 1
                count, tweetId = self.tweetMap[followeeId][idx]
                # add to the minHeap according to count
                heapq.heappush(minHeap, [count, tweetId, followeeId, idx - 1])
            
        while minHeap and len(res) < 10:
            count, tweetId, followeeId, idx = heapq.heappop(minHeap)
            res.append(tweetId)
            if idx >= 0:
                count, tweetId = self.tweetMap[followeeId][idx]
                heapq.heappush(minHeap, [count, tweetId, followeeId, idx - 1])
        
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.followMap[followerId]:
            self.followMap[followerId].remove(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
# @lc code=end

