#
# @lc app=leetcode id=116 lang=python3
#
# [116] Populating Next Right Pointers in Each Node
#

# @lc code=start
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        q = collections.deque([root])
        while q:
            qLen = len(q)
            for i in range(qLen):
                if q[i]:
                    if (i + 1) < qLen and q[i+1]:
                        q[i].next = q[i + 1]
                    else:
                        q[i].next = None
                        
            for i in range(qLen):
                node = q.popleft()
                if node:
                    q.append(node.left)
                    q.append(node.right)
        
        return root


# @lc code=end

