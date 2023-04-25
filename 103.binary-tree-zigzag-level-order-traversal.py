#
# @lc app=leetcode id=103 lang=python3
#
# [103] Binary Tree Zigzag Level Order Traversal
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        q = collections.deque()
        q.append(root)
        res = []
        cnt = 0 # count the current level

        while q:
            cnt += 1
            level = []
            qLen = len(q)
            if cnt % 2:
                for i in range(qLen):
                    # odd level
                    node = q.popleft()
                    if node:
                        level.append(node.val)
                        q.append(node.left)
                        q.append(node.right)
            else:
                for i in range(qLen-1, -1, -1):
                    node = q[i]
                    if node:
                        level.append(node.val)
                for i in range(qLen):
                    node = q.popleft()
                    if node:
                        q.append(node.left)
                        q.append(node.right)
            if level:
                res.append(level)
        
        return res
                



        
# @lc code=end

