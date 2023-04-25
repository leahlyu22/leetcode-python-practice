#
# @lc app=leetcode id=653 lang=python3
#
# [653] Two Sum IV - Input is a BST
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        visited = set()
        
        def dfs(node, visited):
            if not node:
                return False

            target = k - node.val
            if target in visited:
                return True
            visited.add(node.val)
            return dfs(node.left, visited) or dfs(node.right, visited)
        
        return dfs(root, visited)

# @lc code=end

