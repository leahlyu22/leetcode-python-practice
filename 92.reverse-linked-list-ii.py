#
# @lc app=leetcode id=92 lang=python3
#
# [92] Reverse Linked List II
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # first find left and right node
        dummy = ListNode(0, head)
        l = r = dummy
        while left:
            groupPrev = l
            l = l.next
            left -= 1

        while right:
            r = r.next
            right -= 1
        groupNxt = r.next

        # inside group reverse
        prev = groupNxt
        curr = l
        while curr != groupNxt:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        
        # outside group reverse
        groupPrev.next = r

        return dummy.next
        


# @lc code=end

