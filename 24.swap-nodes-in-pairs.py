#
# @lc app=leetcode id=24 lang=python3
#
# [24] Swap Nodes in Pairs
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        pre, cur = dummy, head

        while cur and cur.next:
            nxtPair = cur.next.next
            second = cur.next

            # reverse the current pair
            second.next = cur
            cur.next = nxtPair
            pre.next = second

            # update pair
            pre = cur
            cur = nxtPair
        
        return dummy.next

        
# @lc code=end

