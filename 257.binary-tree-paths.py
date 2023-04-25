#
# @lc app=leetcode id=257 lang=python3
#
# [257] Binary Tree Paths
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        # use dfs
        res = []

        def dfs(node, path):
            if not node:
                return
            
            if path != "":
                path += "->"
            path += str(node.val)
            
            if not node.left and not node.right:
                res.append(path)
            if node.left:
                dfs(node.left, path)
            if node.right:
                dfs(node.right, path)

        dfs(root, "")
        return res
# @lc code=end

