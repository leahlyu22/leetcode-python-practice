#
# @lc app=leetcode id=83 lang=python3
#
# [83] Remove Duplicates from Sorted List
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(None, head)
        prev = dummy
        cur = head

        while cur:
            while cur and prev.val == cur.val:
                cur = cur.next if cur else None
            prev.next = cur
            prev = prev.next
            cur = cur.next if cur else None
            
        return dummy.next 

# @lc code=end

