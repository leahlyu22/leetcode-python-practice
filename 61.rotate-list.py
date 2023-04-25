#
# @lc app=leetcode id=61 lang=python3
#
# [61] Rotate List
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next or k == 0:
            return head
        
        old_head = head
        curr = head
        size = 0 # calculate the size of the list

        while curr:
            curr = curr.next
            size += 1
        
        if k % size == 0:
            return head
        k = k % size

        slow, fast = head, head
        while fast and fast.next:
            if k <= 0:
                slow = slow.next
            fast = fast.next
            k -= 1
        
        new_head, new_tail, old_tail = slow.next, slow, fast
        new_tail.next = None
        old_tail.next = old_head

        return new_head
        

# @lc code=end

