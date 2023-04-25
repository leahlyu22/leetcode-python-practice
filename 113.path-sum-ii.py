#
# @lc app=leetcode id=113 lang=python3
#
# [113] Path Sum II
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        res = []
        
        def dfs(node, cur, curSum):
            if not node:
                return
            # if curSum > targetSum:
            #     return []
            
            curSum += node.val
            cur.append(node.val)

            if not node.left and not node.right and curSum == targetSum:
                res.append(cur.copy())
            
            dfs(node.left, cur, curSum)
            dfs(node.right, cur, curSum)
            cur.pop()
            return cur
        
        dfs(root, [], 0)
        return res
# @lc code=end

