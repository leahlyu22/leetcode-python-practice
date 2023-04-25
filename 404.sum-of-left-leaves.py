#
# @lc app=leetcode id=404 lang=python3
#
# [404] Sum of Left Leaves
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:

        def dfs(node, isLeft):
            if not node:
                return 0
            if not node.left and not node.right and isLeft:
                return node.val

            leftSum = dfs(node.left, True)
            rightSum = dfs(node.right, False)
            return leftSum + rightSum
        
        return dfs(root, False)
        
# @lc code=end

