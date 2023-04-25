#
# @lc app=leetcode id=148 lang=python3
#
# [148] Sort List
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        # the first element is initially sorted
        dummy = ListNode(None, head)
        target = dummy.next.next
        
        while target:
            cur = dummy.next
            prev = dummy
            temp = target.next
            while cur != target:
                if cur.val < target.val:
                    prev = cur
                    cur = cur.next
                else:
                    prev.next = target
                    target.next = cur
                    while cur.next != target:
                        cur = cur.next
                    cur.next = temp
                    break
            target = temp
            
        
        return dummy.next
# @lc code=end

