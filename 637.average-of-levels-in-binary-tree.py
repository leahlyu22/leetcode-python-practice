#
# @lc app=leetcode id=637 lang=python3
#
# [637] Average of Levels in Binary Tree
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        q = collections.deque()
        q.append(root)
        res = []

        while q:
            levelSum = 0
            qLen = len(q)
            cnt = 0
            for i in range(qLen):
                node = q.popleft()
                if node:
                    levelSum += node.val
                    q.append(node.left)
                    q.append(node.right)
                    cnt += 1
            if cnt:
                res.append(levelSum / cnt)
        
        return res

            



# @lc code=end

